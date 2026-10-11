# nodes.py：图的节点。每个函数读 state、返回自己改动的字段
import hashlib, json, re, sys
from datetime import datetime
from pathlib import Path
from langgraph.types import Send, interrupt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prompt-evaluation"))
from src.prompt_builder import load_prompts, build_prompt, SENTENCE_TASKS, KEYWORD_TASK   # 实验仓库
from src.evaluators import parse_keywords                                                  # 实验仓库

import llm
from checks import check_sentence, lexical_overlap, verify_quotes, find_contradictions, unknown_fields
from contract import build_output, validate
from parse_brief import load_brief

RECIPE = json.loads((HERE / "recipe.json").read_text(encoding="utf-8"))
RECIPE_HASH = hashlib.sha1((HERE / "recipe.json").read_bytes()).hexdigest()[:12]   # 配置内容哈希：同名配置改了内容也能发现
llm.RECIPE = RECIPE
TEMPLATES = load_prompts(HERE.parent / "prompt-evaluation" / "prompts.txt")
ALL_TASKS = SENTENCE_TASKS + [KEYWORD_TASK]
PER_TASK_BUDGET_USD = RECIPE["budget_per_brief_usd"] / len(ALL_TASKS)   # 软上限：按栏分摊；调用前检查，最多超出一次调用的费用
CLAR = RECIPE["clarification"]                                          # 轮数上限、哪些字段算关键缺口
CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")                         # 客户编号会拼进路径，只许这些字符


def data_dir(client_id):
    if not CLIENT_ID.match(client_id or ""):
        raise ValueError(f"客户编号不合法：{client_id!r}")
    return HERE / RECIPE["client_isolation"]["data_root"] / client_id


# ---------- 1 读取简报 ----------
def brief_id_for(path):
    """简报编号 = 文件名 + 内容哈希前 8 位：同名但内容不同的文件是两次任务，不会误续旧记录"""
    p = Path(path)
    return f"{p.stem}-{hashlib.sha1(p.read_bytes()).hexdigest()[:8]}"


def source_hash_for(path):
    """当前输入文件的内容哈希：客户补充简报后会变；brief_id（任务身份）不变"""
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]


def read_brief(state):
    brief, missing = load_brief(state["brief_path"])
    return {"brief": brief, "missing_fields": missing,
            "recipe_version": state.get("recipe_version") or RECIPE["recipe_version"],   # 任务启动时固定，恢复与客户补充时沿用
            "recipe_hash": state.get("recipe_hash") or RECIPE_HASH,
            "brief_id": state.get("brief_id") or brief_id_for(state["brief_path"]),   # 任务身份：入口算好，这里只兜底
            "source_hash": source_hash_for(state["brief_path"])}                     # 输入版本：每次读取都重算


# ---------- 2 检查矛盾（整份简报只做一次）----------
def find_conflicts(state):
    return {"contradictions": find_contradictions(state["brief"], RECIPE["contradiction_rules"])}


# ---------- 扇出：对 9 栏各发一个子任务 ----------
def fan_out(state):
    return [Send("extract_one", {**state, "task": t}) for t in ALL_TASKS]


def _source_text(brief, fields):
    return "\n".join(f"{f}: {brief[f]}" for f in fields if brief.get(f))


# ---------- 3–5 一栏：提取 → 格式 → 出处 → 矛盾 → 定栏状态 ----------
def extract_one(state):
    task, brief = state["task"], state["brief"]
    spec = RECIPE["tasks"][task]
    fields = sorted(spec["fields"])
    item = {"status": "调用失败", "value": None, "quotes": [], "evidence": [], "problems": [],
            "cost_usd": 0.0, "latency_s": 0.0, "attempts": 0, "config_id": f'{spec["prompt_version"]}:{"+".join(fields)}'}

    if len(unknown_fields(brief, fields)) == len(fields):             # 所需字段全空：不调用模型
        item.update(status="未知", value=f"未知：简报未提供 {', '.join(fields)}", problems=["简报缺少所需字段"])
        return {"items": {task: item}}

    prompt = build_prompt(task, fields, brief, prompt_version=spec["prompt_version"], templates=TEMPLATES)
    for attempt in (1, 2):                                             # 格式不合格重试一次
        if item["cost_usd"] >= PER_TASK_BUDGET_USD:                    # 这一栏的预算用完
            item["problems"].append("超出本栏预算"); break
        r = llm.call(spec["model"], prompt)
        item["attempts"] = attempt
        item["cost_usd"] += r["cost_usd"]; item["latency_s"] += r["latency_s"]
        if not r["ok"]:
            item["problems"] = [r["error"]]; continue
        if task == KEYWORD_TASK:
            value, _ = parse_keywords(r["parsed"].value)
            problems = [] if len(value) == spec["keyword_count"] else [f"关键词 {len(value)} 个，要求 {spec['keyword_count']} 个"]
        else:
            value = r["parsed"].value.strip()
            problems = check_sentence(value, spec)
        item.update(value=value, quotes=r["parsed"].quotes, problems=problems, status="通过" if not problems else "格式不合格")
        if not problems: break

    if item["status"] == "通过":                                       # 出处核对（确定性）。只能查"原话在不在"，查不了"原话是否支持结论"
        source = _source_text(brief, fields)
        ev = RECIPE["evidence"]
        if task == KEYWORD_TASK:                                       # 逐个关键词：每个词都要在简报里
            found = [k for k in item["value"] if lexical_overlap(k, source) >= 1.0]
            unmatched = [k for k in item["value"] if k not in found]
            item["evidence"] = found
            if unmatched:
                item["status"], item["problems"] = "需人工确认", [f"这些关键词在简报里找不到：{', '.join(unmatched)}"]
        else:
            found, missing = verify_quotes(item["quotes"], source)
            item["evidence"] = found
            if missing:                                                # 有一条假引用就转人工确认，不论其余引用是否真实
                item["status"], item["problems"] = "需人工确认", [f"模型给出的依据在简报里找不到：{missing}"]
            elif not found:
                item["status"], item["problems"] = "需人工确认", ["模型没有给出原文依据"]
            elif task in ev["interpretive_tasks"]:                     # 解读栏：模型的品牌解读，不是客户表述，标"建议"转人工确认
                item["status"], item["problems"] = "建议", ["解读栏：模型从简报推导的建议，请判断是否成立"]
            elif lexical_overlap(item["value"], source) < ev["min_lexical_overlap"]:
                item["status"], item["problems"] = "需人工确认", ["提取内容与原文用词重合很少，可能是推断"]

    for name, a, b, tasks in state.get("contradictions", []):          # 矛盾：受影响的栏转人工确认
        if task in tasks:
            item["status"] = "需人工确认"; item["problems"].append(f"简报有矛盾（{name}）：{a} 与 {b}")
    return {"items": {task: item}}


# ---------- 条件边：只有"关键缺口"或矛盾才走"问客户"支路；其他缺口保留未知，不打扰客户 ----------
def critical_gaps(state):
    return {"missing": [f for f in state["missing_fields"] if f in CLAR["critical_fields"]],
            "contradictions": [f"{n}: {a} vs {b}" for n, a, b, _ in state["contradictions"]]}


def needs_clarification(state):
    g = critical_gaps(state)
    return "draft_clarification" if (g["missing"] or g["contradictions"]) else "human_review"


# ---------- 合成澄清草稿：规则定问什么，模型只写邮件；最多 max_rounds 轮 ----------
def draft_clarification(state):
    rnd, log = state.get("clarification_round", 0), list(state.get("clarification_log", []))
    g = critical_gaps(state)
    optional = [f for f in state["missing_fields"] if f not in CLAR["critical_fields"]]
    if rnd >= CLAR["max_rounds"]:                                      # 规则：满上限不再问，转人工确认
        log.append({"round": rnd, "action": "stop", "reason": f"已问 {rnd} 轮，达到上限", **g})
        return {"clarification_draft": "", "clarification_log": log}
    prompt = ("Write a short, polite email asking the client to complete their brand brief.\n"
              f"Must ask about: missing {g['missing'] or 'nothing'}; contradictions {g['contradictions'] or 'none'}.\n"
              f"Mention only as optional, in one sentence: {optional or 'nothing'}.\nNo preamble.")
    r = llm.call(CLAR["model"], prompt, suffix="")
    draft = r["parsed"].value if r["ok"] else "（草稿生成失败，请手写）"
    log.append({"round": rnd, "action": "draft", "question": draft, "optional": optional, "cost_usd": r["cost_usd"], **g})
    return {"clarification_draft": draft, "clarification_log": log}


def after_draft(state):
    return "ask_client" if state.get("clarification_draft") else "human_review"


# ---------- 中断一：发给客户，等回复。问不问由客户经理定。interrupt 放在第一行：恢复时这个节点会从头重跑 ----------
def ask_client(state):
    answer = interrupt({"kind": "ask_client", "draft": state["clarification_draft"], "missing": state["missing_fields"],
                        "contradictions": state["contradictions"], "round": state.get("clarification_round", 0)})
    if answer.get("action") == "resent":                               # 客户补充后：换简报、清空结果、从头重跑；之前的花费累计
        spent = state.get("prior_cost_usd", 0.0) + sum(i.get("cost_usd", 0) for i in state.get("items", {}).values())
        return {"brief_path": answer["brief_path"], "items": {"__reset__": True}, "contradictions": [], "prior_cost_usd": spent,
                "clarification_round": state.get("clarification_round", 0) + 1}
    return {}


def after_ask(state):
    return "read_brief" if not state.get("items") else "human_review"


# ---------- 中断二：人逐栏核对。界面返回 {栏: {"action": accept | edit | unknown, "value", "evidence"}} ----------
def human_review(state):
    answers = interrupt({"kind": "human_review", "items": state["items"], "clarification_log": state.get("clarification_log", []),
                         "errors": state.get("review_errors", [])})
    confirmed, errors = {}, []
    for task, item in state["items"].items():
        a = answers.get(task) or {}
        action = a.get("action") or ("accept" if item["status"] == "通过" else None)   # "通过"的栏没答就是采用
        if action not in ("accept", "edit", "unknown"):
            errors.append(f"{task} 状态为 {item['status']}，必须明确：采用(accept) / 修改(edit) / 确认未知(unknown)")
            continue
        if action == "accept" and not item.get("value"):
            errors.append(f"{task} 没有内容，不能采用")
            continue
        if action == "edit" and not a.get("value"):
            errors.append(f"{task} 修改后的内容为空")
            continue
        if action == "accept":                                         # 人核实后认为原内容正确，可以原样采用
            value, evidence, src = item["value"], item.get("evidence", []), "brief"
        elif action == "edit":                                         # 改了内容，依据也要重给；依据来源按事实记：原话在简报里就是 brief
            value, evidence = a["value"], a.get("evidence", [])
            found, missing_q = verify_quotes(evidence, _source_text(state["brief"], list(state["brief"].keys())))
            src = a.get("source") or ("brief" if evidence and not missing_q else "human")
        else:
            value, evidence, src = "未知", [], "human"
        if task == KEYWORD_TASK and isinstance(value, str) and action != "unknown":   # "未知"保持字串，不拆成列表
            value = [w.strip() for w in value.split(",") if w.strip()]
        kind = "gap" if value == "未知" else ("interpretation" if task in RECIPE["evidence"]["interpretive_tasks"] else "fact")
        confirmed[task] = {"value": value, "kind": kind, "action": action, "evidence": evidence, "evidence_source": src,
                           "edited_by_human": action != "accept"}
    if errors:   # 不抛错：节点在拿到回答后抛错，这个回答会留在存档里，之后每次恢复都重放它，任务就卡死了
        return {"review_errors": errors}
    return {"confirmed": confirmed, "confirmed_at": datetime.now().strftime("%Y%m%d_%H%M%S"), "review_errors": []}   # 确认时间在这里冻结


def after_review(state):
    return "human_review" if state.get("review_errors") else "deliver"   # 回答有问题 → 回到核对重新问（新的一次中断）


# ---------- 7 交付前检查 → 写文件。幂等：同一份简报重跑覆盖同名文件，不追加 ----------
def deliver(state):
    if state.get("delivered_path"):                                    # 节点被重跑：已交付就不再写
        return {}
    if state.get("recipe_version") != RECIPE["recipe_version"] or state.get("recipe_hash") != RECIPE_HASH:
        raise ValueError("配置在任务进行中变了（版本名或内容）：请换回原配置重跑，或重新开始这份简报")
    stamp = state.get("confirmed_at") or datetime.now().strftime("%Y%m%d_%H%M%S")
    items = {t: {"value": c["value"], "kind": c["kind"], "evidence": c["evidence"], "evidence_source": c["evidence_source"], "status": "已确认",
                 "ai_status": state["items"][t]["status"], "edited_by_human": c["edited_by_human"]}
             for t, c in state["confirmed"].items()}
    doc = build_output(state["brief_id"], state.get("source_hash"), state["client_id"], RECIPE["recipe_version"], stamp,
                       state.get("reviewer", "cli"), items)         # source_hash：交付基于哪个输入版本；reviewer 写入 confirmed_by
    problems = validate(doc, ALL_TASKS, RECIPE["tasks"])
    if problems:   # 不抛错：退回人工核对改正（抛错的话，续跑只会在这里一次次失败）
        return {"review_errors": [f"交付格式不合格，没有交出：{p}" for p in problems]}
    root = data_dir(state["client_id"])
    for sub in ("confirmed", "run_log", "edits_for_eval"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    out = root / "confirmed" / f"{state['brief_id']}.json"
    out.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    cost = state.get("prior_cost_usd", 0.0) + sum(i.get("cost_usd", 0) for i in state["items"].values()) \
           + sum(e.get("cost_usd", 0) for e in state.get("clarification_log", []))
    run_log = {"brief_id": state["brief_id"], "source_hash": state.get("source_hash"), "time": stamp, "recipe_version": RECIPE["recipe_version"],
               "total_cost_usd": round(cost, 6), "clarification_log": state.get("clarification_log", []),
               "items": {t: {k: v for k, v in it.items() if k not in ("value", "quotes")} for t, it in state["items"].items()}}
    (root / "run_log" / f"{state['brief_id']}.json").write_text(json.dumps(run_log, ensure_ascii=False, indent=2), encoding="utf-8")
    edits = [{"brief_id": state["brief_id"], "recipe_version": RECIPE["recipe_version"], "task": t, "action": c["action"],
              "ai_value": state["items"][t].get("value"), "ai_status": state["items"][t]["status"], "confirmed_value": c["value"]}
             for t, c in state["confirmed"].items() if c["edited_by_human"]]
    (root / "edits_for_eval" / f"{state['brief_id']}.json").write_text(json.dumps(edits, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"delivered_path": str(out)}


def after_deliver(state):
    return "human_review" if state.get("review_errors") else "__end__"   # 交付格式不合格 → 退回核对
