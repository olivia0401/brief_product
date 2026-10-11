# evals/trajectory.py：轨迹评估。用按剧本回答的假模型跑整张图，检查 agent 的"决定"对不对：
#   走了哪条路（问不问客户、转不转人工）、每栏贴了什么状态、调了几次模型、花了多少钱。
# 不花钱、结果确定，所以能进 CI。另外每个场景都检查几条"可信不变量"（见 INVARIANTS）。
#
#   python -m evals.trajectory            # 打印结果表，写 evals/reports/trajectory.json；有失败就退出码 1
import json, sys, tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import llm, nodes
from checks import normalise, STOPWORDS, verify_quotes
from graph import build

SPECS, KW = nodes.RECIPE["tasks"], nodes.KEYWORD_TASK
FACT_TASKS = [t for t in nodes.ALL_TASKS if t != KW and t not in nodes.RECIPE["evidence"]["interpretive_tasks"]]

# 合成简报（现场服务行业），每个字段都够长，足以凑出合格的句子和 10 个关键词
BRIEF = {
    "business_category": "Field service management software for asset-heavy industries",
    "product": "A mobile field service platform that schedules technicians, predicts equipment failures from sensor data, "
               "and guides engineers through repair checklists on site with offline access.",
    "audience": "Maintenance managers at utilities, rail operators and manufacturing plants who run large fleets of ageing "
                "assets and cannot afford unplanned downtime or missed inspections.",
    "differentiators": "Predictions explain which sensor readings drove each alert, setup takes days rather than months, "
                       "and pricing is per site instead of per user.",
    "brand_strategy": "Calm, dependable expertise that keeps critical assets running and lets engineers trust every "
                      "recommendation they are given in the field.",
    "personality": "Dependable, precise, plain-spoken, practical",
}
LABEL = {"business_category": "Business category", "product": "Product", "audience": "Audience",
         "differentiators": "Differentiators", "brand_strategy": "Brand strategy", "personality": "Personality"}


def brief_text(overrides=None, drop=()):
    b = {**BRIEF, **(overrides or {})}
    return "\n".join(f"{LABEL[k]}: {v}" for k, v in b.items() if k not in drop)


# ---------- 假模型：按任务给出"好的"或"坏的"回复 ----------
def _words(task, brief):
    return normalise(" ".join(brief[f] for f in sorted(SPECS[task]["fields"]) if brief.get(f))).split()


def _first_field(task, brief):
    return next(brief[f] for f in sorted(SPECS[task]["fields"]) if brief.get(f))


def good(task, brief):
    """合格回复：字数在范围内、用词全来自简报、引用是简报原话"""
    if task == KW:
        seen = []
        for w in _words(task, brief):
            if len(w) > 3 and w not in STOPWORDS and w not in seen:
                seen.append(w)
        return llm.Extraction(value="\n".join(f"{i}. {w}" for i, w in enumerate(seen[:SPECS[KW]["keyword_count"]], 1)), quotes=[])
    s = SPECS[task]
    return llm.Extraction(value=" ".join(_words(task, brief)[:s["min_words"] + 2]),
                          quotes=[" ".join(_first_field(task, brief).split()[:6])])


def too_short(task, brief):
    return llm.Extraction(value="Too short.", quotes=[" ".join(_first_field(task, brief).split()[:6])])


def fake_quote(task, brief):
    return good(task, brief).model_copy(update={"quotes": ["guaranteed zero downtime for every customer"]})


def no_quote(task, brief):
    return good(task, brief).model_copy(update={"quotes": []})


def invented(task, brief):
    """引用是真的，但内容几乎全是简报里没有的词：模型在推断"""
    filler = "revolutionary synergy unlocks transformative quantum paradigm shifts across holistic omnichannel ecosystems globally forever"
    return good(task, brief).model_copy(update={"value": " ".join((filler + " " + filler).split()[:SPECS[task]["min_words"] + 2])})


def invented_keyword(task, brief):
    terms = [l.split(". ", 1)[1] for l in good(task, brief).value.splitlines()]
    terms[-1] = "blockchain"
    return llm.Extraction(value="\n".join(f"{i}. {w}" for i, w in enumerate(terms, 1)), quotes=[])


class Script:
    """llm.call 的替身。behaviour[task] 是一串回复（函数 / "error"），按调用次数取；没写的任务一律 good"""
    def __init__(self, behaviour=None, cost=0.0002):
        self.behaviour, self.cost, self.calls = behaviour or {}, cost, {}

    def __call__(self, model_key, prompt, schema=llm.Extraction, suffix=""):
        task = prompt.split("]]", 1)[0].removeprefix("[[task=") if prompt.startswith("[[task=") else "clarification"
        n = self.calls.get(task, 0)
        self.calls[task] = n + 1
        base = {"usage": {}, "input_tokens": 400, "output_tokens": 60, "cost_usd": self.cost, "latency_s": 0.01, "retries": 0}
        if task == "clarification":
            return {**base, "parsed": llm.Extraction(value="Could you tell us more about your audience?", quotes=[]), "ok": True, "error": None}
        seq = self.behaviour.get(task, [good])
        step = seq[min(n, len(seq) - 1)]
        if step == "error":
            return {**base, "parsed": None, "cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "ok": False,
                    "error": "AuthenticationError: invalid api key"}
        return {**base, "parsed": step(task, self.brief), "ok": True, "error": None}


# ---------- 人的回答 ----------
def review_accept_safe(payload):
    """审核员策略：模型给了内容、且状态是"通过"或"建议"就采用；其余确认未知"""
    return {t: {"action": "accept"} if it.get("value") and it["status"] in ("通过", "建议") else {"action": "unknown"}
            for t, it in payload["items"].items()}


def skip_client(payload):
    return {"action": "skip"}


def resend_same(brief_path):
    return lambda payload: {"action": "resent", "brief_path": brief_path}


# ---------- 场景 ----------
@dataclass
class Scenario:
    name: str
    tests: str                                   # 这个场景在检查什么
    path: list[str]                              # 期望经过的节点（连续重复合并）。"⏸X" = 在 X 停下等人；X 在人回答后才算完成一次
    brief: str = field(default_factory=brief_text)
    behaviour: dict = field(default_factory=dict)
    answers: list = field(default_factory=lambda: [review_accept_safe])   # 每次中断按顺序用一个
    status: dict = field(default_factory=dict)   # 第一次人工核对时，这些栏应有的状态
    calls: dict = field(default_factory=dict)    # 这些任务应调用模型的次数
    problem: dict = field(default_factory=dict)  # 这些栏的问题说明里应包含的字
    cost: float = 0.0002
    extra: Callable | None = None                # 额外检查：result -> 失败说明列表


HAPPY = ["read_brief", "find_conflicts", "extract_one", "⏸human_review", "human_review", "deliver"]
ASK_THEN_REVIEW = ["read_brief", "find_conflicts", "extract_one", "draft_clarification", "⏸ask_client", "ask_client",
                   "⏸human_review", "human_review", "deliver"]

SCENARIOS = [
    Scenario("happy_path", "完整简报、模型回复都合格：事实栏通过，解读栏标建议，直接交付，不问客户",
             HAPPY, status={**{t: "通过" for t in FACT_TASKS}, KW: "通过",
                            **{t: "建议" for t in nodes.RECIPE["evidence"]["interpretive_tasks"]}},
             calls={t: 1 for t in nodes.ALL_TASKS}),
    Scenario("fabricated_quote", "模型编造一句简报里没有的引用：这栏不能通过，转人工",
             HAPPY, behaviour={"function_relevant": [fake_quote]},
             status={"function_relevant": "需人工确认"}, problem={"function_relevant": "找不到"}),
    Scenario("no_evidence", "模型没给任何引用：不能凭空通过",
             HAPPY, behaviour={"category_relevant": [no_quote]},
             status={"category_relevant": "需人工确认"}, problem={"category_relevant": "没有给出原文依据"}),
    Scenario("inferred_content", "引用是真的，但内容是简报里没有的词（模型在推断）：转人工",
             HAPPY, behaviour={"feature_relevant": [invented]},
             status={"feature_relevant": "需人工确认"}, problem={"feature_relevant": "可能是推断"}),
    Scenario("invented_keyword", "10 个关键词里有一个简报里没有：整栏转人工，并点名是哪个词",
             HAPPY, behaviour={KW: [invented_keyword]},
             status={KW: "需人工确认"}, problem={KW: "blockchain"}),
    Scenario("format_retry_recovers", "第一次回复太短，重试一次后合格：通过，且只多调一次",
             HAPPY, behaviour={"context_relevant": [too_short, good]},
             status={"context_relevant": "通过"}, calls={"context_relevant": 2}),
    Scenario("format_fails_twice", "两次都不合格：标格式不合格，最多调两次，不无限重试",
             HAPPY, behaviour={"context_relevant": [too_short]},
             status={"context_relevant": "格式不合格"}, calls={"context_relevant": 2}),
    Scenario("llm_error_isolated", "一栏调用失败：只影响这一栏，其他栏照常，流程照常到人工核对",
             HAPPY, behaviour={"benefit_relevant": ["error"]},
             status={"benefit_relevant": "调用失败", "function_relevant": "通过"}, problem={"benefit_relevant": "AuthenticationError"}),
    Scenario("budget_stops_retry", "单次调用就超出本栏预算：格式不合格时不再重试",
             HAPPY, behaviour={t: [too_short] for t in nodes.ALL_TASKS if t != KW}, cost=0.01,
             status={"context_relevant": "格式不合格"}, calls={"context_relevant": 1}, problem={"context_relevant": "超出本栏预算"}),
    Scenario("contradiction_routes_to_human", "简报同时说高端和最便宜：先起草澄清邮件；受影响的两栏转人工",
             ASK_THEN_REVIEW,
             brief=brief_text({"differentiators": BRIEF["differentiators"] + " A premium service at the cheapest price."}),
             answers=[skip_client, review_accept_safe],
             status={"position_relevant": "需人工确认", "benefit_relevant": "需人工确认", "function_relevant": "通过"},
             problem={"position_relevant": "矛盾"}),
    Scenario("noncritical_gap_no_call", "缺品牌策略（非关键字段）：不打扰客户；只靠这个字段的栏标未知、不调模型",
             HAPPY, brief=brief_text(drop=("brand_strategy",)),
             status={"concept_relevant": "未知", "function_relevant": "通过"}, calls={"concept_relevant": 0}),
    Scenario("critical_gap_asks_client", "缺受众（关键字段）：先起草澄清邮件，停下等客户经理决定",
             ASK_THEN_REVIEW,
             brief=brief_text(drop=("audience",)), answers=[skip_client, review_accept_safe],
             status={"context_relevant": "未知"}, calls={"context_relevant": 0}),
    Scenario("clarification_capped", "客户回复了但还是没补：只问一轮，第二轮停止并转人工",
             ["read_brief", "find_conflicts", "extract_one", "draft_clarification", "⏸ask_client", "ask_client",
              "read_brief", "find_conflicts", "extract_one", "draft_clarification", "⏸human_review", "human_review", "deliver"],
             brief=brief_text(drop=("audience",)), answers=["RESEND", review_accept_safe],
             extra=lambda r: [] if [e["action"] for e in r["state"].get("clarification_log", [])] == ["draft", "stop"]
             else [f"澄清记录应是 draft→stop，实际 {[e['action'] for e in r['state'].get('clarification_log', [])]}"]),
    Scenario("incomplete_review_asks_again", "审核员漏答一栏需确认的：不报错，带着说明重新停下；补答后交付",
             ["read_brief", "find_conflicts", "extract_one", "⏸human_review", "human_review", "⏸human_review",
              "human_review", "deliver"],
             behaviour={"concept_relevant": [no_quote]},
             answers=[lambda p: {t: a for t, a in review_accept_safe(p).items() if t != "concept_relevant"}, review_accept_safe]),
]


# ---------- 可信不变量：每个场景都要成立 ----------
def _inv_pass_has_evidence(r):
    """标"通过"的栏必须有核实过的原文依据（关键词栏：每个词都在简报里）"""
    out = []
    for t, it in r["first_review"].items():
        if it["status"] == "通过":
            if not it.get("evidence"):
                out.append(f"{t} 标通过但没有依据")
            elif t != KW and verify_quotes(it["evidence"], r["brief_text"])[1]:
                out.append(f"{t} 的依据不在简报里")
    return out


def _inv_bounded_calls(r):
    """每栏最多调两次模型；整份简报的花费不超过预算加一次调用"""
    out = [f"{t} 调了 {n} 次" for t, n in r["calls"].items() if t != "clarification" and n > 2]
    budget = nodes.RECIPE["budget_per_brief_usd"]
    if r["state"].get("delivered_path"):
        log = json.loads(Path(r["run_log"]).read_text(encoding="utf-8"))
        if log["total_cost_usd"] > budget + r["max_call_cost"] * len(nodes.ALL_TASKS):
            out.append(f"花费 {log['total_cost_usd']} 超出预算 {budget}")
    return out


def _inv_delivery_confirmed(r):
    """交付的每一栏都经过人确认；没有交付就不能有交付文件"""
    if not r["state"].get("delivered_path"):
        return ["没有交付"]
    doc = json.loads(Path(r["state"]["delivered_path"]).read_text(encoding="utf-8"))
    return [f"{t} 未确认就交付" for t, it in doc["items"].items() if it["status"] != "已确认"]


INVARIANTS = {"pass_has_evidence": _inv_pass_has_evidence, "bounded_calls": _inv_bounded_calls,
              "delivered_only_after_review": _inv_delivery_confirmed}


# ---------- 运行 ----------
def _collapse(path):
    return [p for i, p in enumerate(path) if i == 0 or p != path[i - 1]]


def run_scenario(sc: Scenario, workdir: Path) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    brief_path = workdir / f"{sc.name}.txt"
    brief_path.write_text(sc.brief, encoding="utf-8")
    script = Script(sc.behaviour, sc.cost)
    script.brief = nodes.load_brief(brief_path)[0]                    # 和 agent 读到的字段一致

    saved = (llm.call, nodes.build_prompt, nodes.HERE)
    real_build_prompt = nodes.build_prompt
    llm.call = script
    nodes.build_prompt = lambda task, *a, **k: f"[[task={task}]]" + real_build_prompt(task, *a, **k)   # 让假模型知道是哪一栏
    nodes.HERE = workdir                                                                              # 记录写到临时目录
    try:
        cfg = {"configurable": {"thread_id": sc.name}}
        path, first_review, answers = [], None, list(sc.answers)
        inp = {"client_id": "eval", "brief_path": str(brief_path), "brief_id": nodes.brief_id_for(brief_path)}
        with SqliteSaver.from_conn_string(str(workdir / "ck.db")) as saver:
            app = build(saver)
            for _ in range(10):                                                       # 防止流程打转
                for chunk in app.stream(inp, cfg, stream_mode="updates", durability="sync"):
                    path += [k for k in chunk if k != "__interrupt__"]
                snap = app.get_state(cfg)
                if not snap.interrupts:
                    break
                payload = snap.interrupts[0].value
                path.append(f"⏸{payload['kind']}")
                if payload["kind"] == "human_review" and first_review is None:
                    first_review = payload["items"]
                if not answers:
                    break
                a = answers.pop(0)
                inp = Command(resume=resend_same(str(brief_path))(payload) if a == "RESEND" else a(payload))
            state = snap.values
    finally:
        llm.call, nodes.build_prompt, nodes.HERE = saved

    r = {"state": state, "path": _collapse(path), "calls": script.calls, "first_review": first_review or {},
         "brief_text": nodes._source_text(script.brief, list(script.brief)), "max_call_cost": sc.cost,
         "run_log": workdir / "data" / "eval" / "run_log" / f"{state.get('brief_id')}.json"}
    fails = []
    if r["path"] != sc.path:
        fails.append(f"路径不对：期望 {' → '.join(sc.path)}；实际 {' → '.join(r['path'])}")
    for t, want in sc.status.items():
        got = r["first_review"].get(t, {}).get("status")
        if got != want:
            fails.append(f"{t} 状态应为 {want}，实际 {got}")
    for t, want in sc.calls.items():
        if script.calls.get(t, 0) != want:
            fails.append(f"{t} 应调用模型 {want} 次，实际 {script.calls.get(t, 0)} 次")
    for t, text in sc.problem.items():
        if not any(text in p for p in r["first_review"].get(t, {}).get("problems", [])):
            fails.append(f"{t} 的问题说明里应包含「{text}」，实际 {r['first_review'].get(t, {}).get('problems')}")
    if sc.extra:
        fails += sc.extra(r)
    inv = {name: f(r) for name, f in INVARIANTS.items()}
    fails += [f"[不变量 {name}] {m}" for name, ms in inv.items() for m in ms]
    return {"scenario": sc.name, "tests": sc.tests, "passed": not fails, "failures": fails, "path": r["path"],
            "llm_calls": sum(n for t, n in script.calls.items()), "invariants": {k: not v for k, v in inv.items()}}


def run_all(workdir: Path) -> dict:
    results = []
    for sc in SCENARIOS:
        try:
            results.append(run_scenario(sc, workdir / sc.name))
        except Exception as e:                                   # 场景跑崩也是失败，不让整个评估中断
            results.append({"scenario": sc.name, "tests": sc.tests, "passed": False, "path": [], "llm_calls": 0,
                            "failures": [f"运行出错：{type(e).__name__}: {e}"], "invariants": {}})
    status_checks = sum(len(sc.status) for sc in SCENARIOS)
    status_ok = status_checks - sum(1 for res in results for f in res["failures"] if "状态应为" in f)
    return {"scenarios": len(results), "passed": sum(r["passed"] for r in results),
            "path_correct": sum(not any(f.startswith("路径不对") for f in r["failures"]) for r in results),
            "status_correct": f"{status_ok}/{status_checks}",
            "invariant_violations": sum(1 for r in results for f in r["failures"] if f.startswith("[不变量")),
            "results": results}


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        report = run_all(Path(d))
    out = Path(__file__).resolve().parent / "reports" / "trajectory.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report["results"]:
        print(f"{'PASS' if r['passed'] else 'FAIL'}  {r['scenario']:<32} 模型调用 {r['llm_calls']:>2}  {r['tests']}")
        for f in r["failures"]:
            print(f"        - {f}")
    print(f"\n场景 {report['passed']}/{report['scenarios']} 通过 · 路径正确 {report['path_correct']}/{report['scenarios']} · "
          f"栏状态正确 {report['status_correct']} · 不变量违反 {report['invariant_violations']}")
    print(f"报告：{out}")
    sys.exit(0 if report["passed"] == report["scenarios"] else 1)
