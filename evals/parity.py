# evals/parity.py：生产一致性评估。回答两个问题：
#   1. 上线版和实验版质量一样吗？同一批真实简报、同一时间、同一模型，跑两组，配对比较：
#        实验组 = 实验仓库自己的客户端 + 原 prompt + 纯文本回答（实验当时的代码路径）
#        生产组 = 完整的 agent（结构化输出 + 要求给原话 + 格式重试 + 核对 + 路由）
#      分数用实验自己的打分函数（余弦 / 关键词 F1），容差 = recipe 里记录的决策容差。
#   2. agent 贴的"通过"标签可信吗？"通过"的栏里多少真的合格；不合格的栏里多少被拦下转人工。
#
# 简报和参考答案是客户数据：只在本地读（实验仓库的 briefs.yml），逐条结果写进 evals/reports/（不进 git），
# 对外只用汇总数字（parity_summary.json）。
#
#   python -m evals.parity --limit 2      # 先试跑 2 份
#   python -m evals.parity                # 全部
import argparse, json, random, statistics, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

import llm, nodes                                   # nodes 会把实验仓库加进 sys.path
from graph import build
from src import config as exp_cfg                   # 实验仓库的配置：导入时从它的 .env 读 OPENAI_API_KEY
from src.llm_client import LLMClient, DONE_STATUSES
from src.evaluators import EmbeddingClient, score_sentence, score_keywords, parse_keywords
from src.prompt_builder import load_briefs, build_prompt, SENTENCE_TASKS, KEYWORD_TASK

HERE = Path(__file__).resolve().parent
REPORTS = HERE / "reports"
INPUT_FIELDS = ["business_category", "product", "differentiators", "audience", "brand_strategy", "personality"]
TOLERANCE = nodes.RECIPE["provenance"]["decision_tolerance_cosine"]     # 0.036（= 2×运行噪声 SD 0.018）：实验里"差异小于测量噪声"的界线
USABLE = 0.60                                                           # 实验工作簿的分档：≥0.60 可用，<0.60 需仔细复核
CAP_PROD_USD, CAP_EXP_USD = 0.60, 0.30                                  # 本次评估的花费硬上限
# 实验当时的分数：Results/Prompt Eval Results.xlsx，Final Recommendations 第 3 节 "Avg score"（同一 recipe，Stage A+B，23 份）
HISTORICAL = {"benefit_relevant": 0.627, "category_relevant": 0.672, "concept_relevant": 0.652, "context_relevant": 0.614,
              "emotion_relevant": 0.604, "feature_relevant": 0.732, "function_relevant": 0.759, "position_relevant": 0.677}
INTERPRETIVE = set(nodes.RECIPE["evidence"]["interpretive_tasks"])
FACT_TASKS = [t for t in SENTENCE_TASKS if t not in INTERPRETIVE]


# ---------- 实验组 ----------
def run_experiment_arm(briefs, workers=8):
    client = LLMClient.from_env(budget_cap_usd=CAP_EXP_USD)
    jobs = [(i, b, t) for i, b in enumerate(briefs) for t in nodes.ALL_TASKS]

    def one(job):
        i, b, t = job
        spec = nodes.RECIPE["tasks"][t]
        prompt = build_prompt(task=t, fields=tuple(sorted(spec["fields"])), brief=b,
                              prompt_version=spec["prompt_version"], templates=nodes.TEMPLATES)
        r = client.call(model_key=spec["model"], prompt=prompt, brief_id=f"b{i:02d}", task=t, config_id="parity", run_id=1)
        ok = r.status in DONE_STATUSES and (r.raw_response or "").strip()          # 和实验 analyze.py 一样：长度超标的也打分
        return (i, t), {"value": (r.raw_response or "").strip() if ok else None, "cost_usd": r.cost_usd,
                        "status": r.status, "latency_s": r.latency_s}

    with ThreadPoolExecutor(workers) as ex:
        out = dict(ex.map(one, jobs))
    return out, client.total_cost_usd


# ---------- 生产组 ----------
class CappedCall:
    """包住 llm.call：累计花费超过上限就不再真的调用（返回失败），防止评估失控"""
    def __init__(self, real, cap):
        self.real, self.cap, self.spent, self.lock = real, cap, 0.0, threading.Lock()

    def __call__(self, *a, **k):
        with self.lock:
            if self.spent >= self.cap:
                return {"parsed": None, "usage": {}, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                        "latency_s": 0.0, "retries": 0, "ok": False, "error": "EvalBudgetExceeded"}
        r = self.real(*a, **k)
        with self.lock:
            self.spent += r["cost_usd"]
        return r


def run_production_brief(i, brief, workdir):
    """跑完整张图，停在第一次人工核对，取出 agent 交给人的每一栏（内容 + 状态）。澄清一律"不问客户"，以便无人值守"""
    path = workdir / f"b{i:02d}.json"
    path.write_text(json.dumps({f: brief[f] for f in INPUT_FIELDS if f in brief}, ensure_ascii=False), encoding="utf-8")
    cfg = {"configurable": {"thread_id": f"parity-{i}"}}
    t0, inp, asked = time.time(), {"client_id": "parity", "brief_path": str(path), "brief_id": nodes.brief_id_for(path)}, 0
    with SqliteSaver.from_conn_string(str(workdir / f"ck{i:02d}.db")) as saver:
        app = build(saver)
        for _ in range(5):
            app.invoke(inp, cfg, durability="sync")
            snap = app.get_state(cfg)
            payload = snap.interrupts[0].value if snap.interrupts else None
            if payload is None or payload["kind"] == "human_review":
                break
            asked += 1
            inp = Command(resume={"action": "skip"})
    items = (payload or {}).get("items", {})
    clar_cost = sum(e.get("cost_usd", 0) for e in snap.values.get("clarification_log", []))
    return i, {"items": items, "asked_client": asked, "wall_s": time.time() - t0,
               "cost_usd": sum(it.get("cost_usd", 0) for it in items.values()) + clar_cost}


def run_production_arm(briefs, workers=4):
    capped = CappedCall(llm.call, CAP_PROD_USD)
    saved = (llm.call, nodes.HERE)
    llm.call = capped
    try:
        with tempfile.TemporaryDirectory() as d:
            nodes.HERE = Path(d)                                         # 万一写记录，写到临时目录
            with ThreadPoolExecutor(workers) as ex:
                out = dict(ex.map(lambda ib: run_production_brief(ib[0], ib[1], Path(d)), enumerate(briefs)))
    finally:
        llm.call, nodes.HERE = saved
    return out, capped.spent


def refresh_task(task, briefs, prod, workdir):
    """修了某一栏的代码后，只重跑这一栏（直接调生产的 extract_one，与图里同一段代码），换掉存档里的旧结果。
    前提：所有简报都没走"问客户"支路（无矛盾、无关键缺口），这一栏的输入才和整图运行时一样"""
    assert all(p["asked_client"] == 0 for p in prod.values()), "有简报走过澄清支路，不能只重跑单栏"
    capped, saved = CappedCall(llm.call, 0.10), llm.call
    llm.call = capped
    try:
        def one(ib):
            i, b = ib
            path = workdir / f"b{i:02d}.json"
            path.write_text(json.dumps({f: b[f] for f in INPUT_FIELDS if f in b}, ensure_ascii=False), encoding="utf-8")
            brief, _ = nodes.load_brief(path)
            return i, nodes.extract_one({"task": task, "brief": brief, "contradictions": []})["items"][task]
        with ThreadPoolExecutor(8) as ex:
            new = dict(ex.map(one, enumerate(briefs)))
    finally:
        llm.call = saved
    delta = 0.0
    for i, item in new.items():
        d = item.get("cost_usd", 0) - prod[i]["items"][task].get("cost_usd", 0)
        prod[i]["cost_usd"] += d; delta += d
        prod[i]["items"][task] = item
    return capped.spent, delta


# ---------- 打分 ----------
def produced(task, value):
    """这一栏有没有交出内容（关键词：至少解析出一个词）"""
    if value is None or value == "" or value == [] or (isinstance(value, str) and value.startswith("未知")):
        return False
    return bool(value if isinstance(value, list) else parse_keywords(value)[0]) if task == KEYWORD_TASK else True


def score(task, value, brief, emb):
    """和实验 analyze.py 同一套：句子任务 = 与参考答案的余弦；关键词 = F1。
    关键词一个没交出算 F1=0（两组同样处理）：跳过的话就只比了成功的那些。句子没有内容 = None，另报交出率"""
    if task == KEYWORD_TASK:
        if not produced(task, value):
            return 0.0
        terms = value if isinstance(value, list) else parse_keywords(value)[0]
        return score_keywords(terms, brief.get("keywords", []) or []).f1
    if not produced(task, value):
        return None
    return score_sentence(value, brief[task], task, emb).cosine


# ---------- 统计 ----------
def bootstrap_ci(diffs_by_brief, n=5000, seed=0):
    """按简报重抽样（同一份简报的各栏不独立，不能当成独立样本）"""
    rng, keys = random.Random(seed), list(diffs_by_brief)
    means = []
    for _ in range(n):
        sample = [d for k in (rng.choice(keys) for _ in keys) for d in diffs_by_brief[k]]
        means.append(statistics.fmean(sample))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def wilcoxon_p(diffs):
    from scipy.stats import wilcoxon
    nz = [d for d in diffs if abs(d) > 1e-12]
    return float(wilcoxon(nz).pvalue) if len(nz) >= 6 else None


def verdict(mean, lo, hi):
    """非劣效判断：上线最怕的是变差，所以先看置信区间下限有没有低于 -容差"""
    if hi < -TOLERANCE:
        return "变差（超出容差）"
    if lo < -TOLERANCE:
        return "不能排除变差"
    if hi <= TOLERANCE:
        return "一致（差异在容差内）"
    if lo > TOLERANCE:
        return "变好（超出容差）"
    return "不差于实验版（可能更好）"


def router_metrics(rows, tasks, threshold):
    rs = [r for r in rows if r["task"] in tasks and r["prod_score"] is not None]
    passed = [r for r in rs if r["prod_status"] == "通过"]
    bad = [r for r in rs if r["prod_score"] < threshold]
    good_flagged = [r for r in rs if r["prod_status"] != "通过" and r["prod_score"] >= threshold]
    return {"threshold": threshold, "n": len(rs), "n_pass": len(passed), "n_bad": len(bad),
            "base_good_rate": round(1 - len(bad) / len(rs), 3) if rs else None,     # 不看标签、随便拿一栏合格的概率
            "pass_precision": round(sum(r["prod_score"] >= threshold for r in passed) / len(passed), 3) if passed else None,
            "catch_recall": round(sum(r["prod_status"] != "通过" for r in bad) / len(bad), 3) if bad else None,
            "good_but_flagged": len(good_flagged)}


def auc(pos, neg):
    """随机取一个"通过"栏和一个被拦栏，"通过"那栏分数更高的概率；0.5 = 标签和质量无关"""
    if not pos or not neg:
        return None
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)


def summarise(rows, prod, exp_cost, prod_cost, n_briefs):
    per_task = {}
    for t in nodes.ALL_TASKS:
        rs = [r for r in rows if r["task"] == t and r["prod_score"] is not None and r["exp_score"] is not None]
        if not rs:
            continue
        by_brief = {r["brief"]: [r["prod_score"] - r["exp_score"]] for r in rs}
        lo, hi = bootstrap_ci(by_brief)
        diffs = [r["prod_score"] - r["exp_score"] for r in rs]
        all_t = [r for r in rows if r["task"] == t]
        per_task[t] = {"metric": "f1" if t == KEYWORD_TASK else "cosine", "n": len(rs),
                       "experiment_coverage": round(sum(r["exp_produced"] for r in all_t) / len(all_t), 3),
                       "production_coverage": round(sum(r["prod_produced"] for r in all_t) / len(all_t), 3),
                       "experiment_arm": round(statistics.fmean(r["exp_score"] for r in rs), 3),
                       "production_arm": round(statistics.fmean(r["prod_score"] for r in rs), 3),
                       "historical": HISTORICAL.get(t), "diff": round(statistics.fmean(diffs), 3),
                       "ci95": [round(lo, 3), round(hi, 3)], "wilcoxon_p": wilcoxon_p(diffs),
                       "verdict": verdict(statistics.fmean(diffs), lo, hi)}
    sent = [r for r in rows if r["task"] in SENTENCE_TASKS and r["prod_score"] is not None and r["exp_score"] is not None]
    by_brief = {}
    for r in sent:
        by_brief.setdefault(r["brief"], []).append(r["prod_score"] - r["exp_score"])
    lo, hi = bootstrap_ci(by_brief)
    overall_diff = statistics.fmean(d for ds in by_brief.values() for d in ds)

    fact_rows = [r for r in rows if r["task"] in FACT_TASKS and r["prod_score"] is not None]
    statuses = {}
    for r in rows:
        s = statuses.setdefault(r["prod_status"], {"n": 0, "scores": []})
        s["n"] += 1
        if r["prod_score"] is not None and r["task"] in SENTENCE_TASKS:
            s["scores"].append(r["prod_score"])
    walls = sorted(p["wall_s"] for p in prod.values())
    items = [it for p in prod.values() for it in p["items"].values()]
    return {
        "briefs": n_briefs, "tolerance": TOLERANCE, "usable_threshold": USABLE,
        "overall_sentence": {"experiment_arm": round(statistics.fmean(r["exp_score"] for r in sent), 3),
                             "production_arm": round(statistics.fmean(r["prod_score"] for r in sent), 3),
                             "diff": round(overall_diff, 3), "ci95": [round(lo, 3), round(hi, 3)],
                             "verdict": verdict(overall_diff, lo, hi)},
        "per_task": per_task,
        "router_fact_tasks": [router_metrics(rows, FACT_TASKS, th) for th in (0.55, 0.60, 0.65, 0.70)],
        "router_auc_fact_tasks": auc([r["prod_score"] for r in fact_rows if r["prod_status"] == "通过"],
                                     [r["prod_score"] for r in fact_rows if r["prod_status"] != "通过"]),
        "status_distribution": {s: {"n": v["n"], "mean_cosine": round(statistics.fmean(v["scores"]), 3) if v["scores"] else None}
                                for s, v in sorted(statuses.items(), key=lambda kv: -kv[1]["n"])},
        "ops": {"production_cost_usd": round(prod_cost, 4), "experiment_cost_usd": round(exp_cost, 4),
                "production_cost_per_brief_usd": round(prod_cost / n_briefs, 4),
                "experiment_cost_per_brief_usd": round(exp_cost / n_briefs, 4),
                "wall_s_p50": round(walls[len(walls) // 2], 1), "wall_s_max": round(walls[-1], 1),
                "input_tokens": sum(it.get("input_tokens", 0) for it in items),
                "output_tokens": sum(it.get("output_tokens", 0) for it in items),
                "transient_retries": sum(it.get("retries", 0) for it in items),
                "format_retries": sum(1 for it in items if it.get("attempts", 0) > 1),
                "briefs_where_agent_wanted_to_ask_client": sum(1 for p in prod.values() if p["asked_client"])},
    }


def print_summary(s):
    o = s["overall_sentence"]
    print(f"\n== 上线版 vs 实验版（{s['briefs']} 份真实简报，配对）==")
    print(f"8 个句子任务合计：实验组 {o['experiment_arm']}  生产组 {o['production_arm']}  差 {o['diff']:+.3f} "
          f"95%CI [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}]  → {o['verdict']}")
    print(f"\n{'任务':<20}{'指标':<7}{'历史':>7}{'实验组':>8}{'生产组':>8}{'差':>8}  {'95%CI':<18}{'p':>7}  交出率(实验/生产)  结论")
    for t, r in s["per_task"].items():
        h = f"{r['historical']:.3f}" if r["historical"] else "  -  "
        p = f"{r['wilcoxon_p']:.3f}" if r["wilcoxon_p"] is not None else "  -  "
        print(f"{t:<20}{r['metric']:<7}{h:>7}{r['experiment_arm']:>8.3f}{r['production_arm']:>8.3f}{r['diff']:>+8.3f}  "
              f"[{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}]  {p:>7}  {r['experiment_coverage']:.2f}/{r['production_coverage']:.2f}        {r['verdict']}")
    print("\n== '通过'标签可信吗（5 个事实栏；合格 = 与参考答案余弦 ≥ 阈值）==")
    for m in s["router_fact_tasks"]:
        print(f"阈值 {m['threshold']:.2f}：整体合格率 {m['base_good_rate']}；通过 {m['n_pass']}/{m['n']} 栏，其中合格 {m['pass_precision']}；"
              f"不合格 {m['n_bad']} 栏，被拦下 {m['catch_recall']}；合格却被拦 {m['good_but_flagged']} 栏")
    print(f"区分度 AUC（通过栏分数高于被拦栏的概率）：{s['router_auc_fact_tasks']}")
    print("\n栏状态分布：" + "  ".join(f"{k} {v['n']}（均分 {v['mean_cosine']}）" for k, v in s["status_distribution"].items()))
    print("运行：" + json.dumps(s["ops"], ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="只跑前 N 份（试跑用）")
    ap.add_argument("--rescore", action="store_true", help="不调模型，用上次存下的两组输出重新打分")
    ap.add_argument("--refresh-task", help="和 --rescore 一起用：只重跑生产组的这一栏（改了这一栏的代码后）")
    a = ap.parse_args()
    briefs = load_briefs(nodes.EXPERIMENT_DIR / "briefs.yml")[: a.limit]
    REPORTS.mkdir(exist_ok=True)
    print(f"{len(briefs)} 份简报 × {len(nodes.ALL_TASKS)} 栏。生产组上限 ${CAP_PROD_USD}，实验组上限 ${CAP_EXP_USD}")

    raw_path = REPORTS / "parity_raw.json"                                 # 模型输出先落盘：打分出错不必重花钱（含客户内容，只留本地）
    if a.rescore:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        prod = {int(k): v for k, v in raw["prod"].items()}
        exp = {(int(k.split("|")[0]), k.split("|")[1]): v for k, v in raw["exp"].items()}
        prod_cost, exp_cost = raw["prod_cost"], raw["exp_cost"]
        assert len(prod) == len(briefs), "存档的简报数和这次不一致"
        if a.refresh_task:
            with tempfile.TemporaryDirectory() as d:
                spent, delta = refresh_task(a.refresh_task, briefs, prod, Path(d))
            prod_cost += delta
            print(f"重跑生产组 {a.refresh_task}：花费 ${spent:.4f}")
            raw.update(prod=prod, prod_cost=prod_cost, refreshed=raw.get("refreshed", []) + [a.refresh_task])
            raw_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    else:
        t0 = time.time()
        prod, prod_cost = run_production_arm(briefs)
        print(f"生产组完成：${prod_cost:.4f}，{time.time() - t0:.0f}s")
        t0 = time.time()
        exp, exp_cost = run_experiment_arm(briefs)
        print(f"实验组完成：${exp_cost:.4f}，{time.time() - t0:.0f}s")
        raw_path.write_text(json.dumps({"prod": prod, "exp": {f"{i}|{t}": v for (i, t), v in exp.items()},
                                        "prod_cost": prod_cost, "exp_cost": exp_cost}, ensure_ascii=False), encoding="utf-8")

    emb = EmbeddingClient(api_key=exp_cfg.OPENAI_API_KEY, model=exp_cfg.EMBEDDING_MODEL,
                          cache_path=REPORTS / "embedding_cache.jsonl")    # 缓存放本仓库的报告目录，不碰实验仓库
    rows = []
    for i, b in enumerate(briefs):
        for t in nodes.ALL_TASKS:
            it = prod[i]["items"].get(t, {})
            rows.append({"brief": i, "task": t, "prod_status": it.get("status", "缺失"),
                         "prod_value": it.get("value"), "exp_value": exp[(i, t)]["value"],
                         "prod_score": score(t, it.get("value"), b, emb), "exp_score": score(t, exp[(i, t)]["value"], b, emb),
                         "prod_produced": produced(t, it.get("value")), "exp_produced": produced(t, exp[(i, t)]["value"]),
                         "prod_problems": it.get("problems", []), "prod_attempts": it.get("attempts")})
    emb.save_cache()
    with (REPORTS / "parity_rows.jsonl").open("w", encoding="utf-8") as f:          # 含客户内容：只留本地
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    s = summarise(rows, prod, exp_cost, prod_cost, len(briefs))
    (REPORTS / "parity_summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(s)


if __name__ == "__main__":
    sys.exit(main())
