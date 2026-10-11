# evals/checkers.py：给"通过"标签找一个能看出内容对不对的检查员。三种候选，在同一批数据上比：
#   C 原文相似度：提取结果和简报原文（这一栏用到的字段）的语义相似度          —— 几乎免费
#   A 自洽性：同一栏用生产代码再答 2 次，三次答案彼此越像越可信                 —— 每栏多 2 次调用
#   B AI 审核员：另起一次调用，看原指令 + 简报 + 答案，打 1–5 分                —— 每栏多 1 次调用
# 用生产一致性评估（evals/parity.py）存下的 23 份真实简报的生产组输出。"不合格" = 与参考答案余弦 < 0.60。
# 检查员看不到参考答案；参考答案只用来打分。
#
#   python -m evals.checkers               # 调模型（结果存盘，重跑不再花钱）
import json, random, statistics, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field
from scipy.stats import spearmanr

import llm, nodes
from checks import lexical_overlap
from src import config as exp_cfg
from src.evaluators import EmbeddingClient, _cosine
from src.prompt_builder import load_briefs
from evals.parity import FACT_TASKS, INPUT_FIELDS, USABLE, CappedCall, REPORTS
from parse_brief import _as_text

BUDGET = 0.30            # 人工审核量：最多把 30% 的栏转给人看
CAP_USD = 0.25           # 本次花费硬上限
N_SAMPLES = 2            # 自洽性：额外再答几次


class Judgement(BaseModel):
    score: int = Field(description="1-5 rating as defined in the instructions")
    reason: str = Field(description="One short sentence explaining the rating")


JUDGE_PROMPT = """You are reviewing one field that an AI extracted from a client's brand brief.

The extractor was given these instructions (they include the relevant parts of the brief):
<instructions>
{prompt}
</instructions>

It answered:
<answer>
{answer}
</answer>

Rate the answer from 1 to 5:
5 = captures exactly what the brief says for this field; nothing invented; nothing important missing
4 = right, with a small omission or slightly generic wording
3 = partly right, or so generic it could describe many brands
2 = mostly misses what the brief says for this field
1 = wrong, off-topic, or not supported by the brief
Judge only against the brief and the instructions, not your own view of the brand."""


def load_items():
    raw = json.loads((REPORTS / "parity_raw.json").read_text(encoding="utf-8"))
    rows = {(r["brief"], r["task"]): r for r in map(json.loads, (REPORTS / "parity_rows.jsonl").open(encoding="utf-8"))}
    briefs = load_briefs(nodes.EXPERIMENT_DIR / "briefs.yml")
    items = []
    for i, b in enumerate(briefs):
        brief = {f: _as_text(b[f]) for f in INPUT_FIELDS if f in b}       # 和生产读到的一样
        for t in FACT_TASKS:
            it = raw["prod"][str(i)]["items"][t]
            spec = nodes.RECIPE["tasks"][t]
            fields = sorted(spec["fields"])
            items.append({"brief": i, "task": t, "value": it["value"], "status": it["status"],
                          "cosine_to_gold": rows[(i, t)]["prod_score"],
                          "source": nodes._source_text(brief, fields),
                          "prompt": nodes.build_prompt(t, fields, brief, prompt_version=spec["prompt_version"],
                                                       templates=nodes.TEMPLATES),
                          "model": spec["model"]})
    return items


# ---------- 三个检查员：分数越高 = 越可信 ----------
def run_checkers(items, emb):
    capped, saved = CappedCall(llm.call, CAP_USD), llm.call
    llm.call = capped

    def one(it):
        out = {}
        out["C_source_similarity"] = _cosine(emb.embed(it["value"]), emb.embed(it["source"]))
        samples = []
        for _ in range(N_SAMPLES):                                       # A：生产代码原样再答（同 prompt、同要求给原话）
            r = llm.call(it["model"], it["prompt"])
            if r["ok"] and r["parsed"].value.strip():
                samples.append(r["parsed"].value.strip())
        answers = [it["value"]] + samples
        vecs = [emb.embed(a) for a in answers]
        pairs = [(i, j) for i in range(len(vecs)) for j in range(i + 1, len(vecs))]
        out["A_self_consistency"] = statistics.fmean(_cosine(vecs[i], vecs[j]) for i, j in pairs) if pairs else None
        r = llm.call(it["model"], JUDGE_PROMPT.format(prompt=it["prompt"], answer=it["value"]), schema=Judgement, suffix="")
        out["B_ai_judge"] = r["parsed"].score if r["ok"] else None
        out["B_reason"] = r["parsed"].reason if r["ok"] else r["error"]
        out["current_overlap"] = lexical_overlap(it["value"], it["source"])   # 现有检查的连续分数，作对照
        return out

    try:
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(one, items))
    finally:
        llm.call = saved
    return results, capped.spent


# ---------- 指标 ----------
def auc(scores, bad):
    """随机取一栏合格的、一栏不合格的，合格那栏检查员分数更高的概率。0.5 = 和瞎猜一样，1 = 完美区分"""
    good_s = [s for s, b in zip(scores, bad) if not b]
    bad_s = [s for s, b in zip(scores, bad) if b]
    return sum((g > x) + 0.5 * (g == x) for g in good_s for x in bad_s) / (len(good_s) * len(bad_s))


def recall_at_budget(scores, bad, budget=BUDGET, shuffles=500, seed=0):
    """把分数最低（最可疑）的 budget 比例转人工，不合格的栏里有多少被拦下。分数相同时随机排，取平均"""
    k, n_bad, rng, rec = round(budget * len(scores)), sum(bad), random.Random(seed), []
    for _ in range(shuffles):
        order = sorted(range(len(scores)), key=lambda i: (scores[i], rng.random()))
        rec.append(sum(bad[i] for i in order[:k]) / n_bad)
    return statistics.fmean(rec)


def leave_one_brief_out(items, scores, bad, budget=BUDGET):
    """更接近上线的做法：阈值只用其他 22 份简报定（它们分数的 30% 分位），再用到留出的那份上。避免在同一批数据上调阈值"""
    flagged = [False] * len(items)
    for b in {it["brief"] for it in items}:
        others = sorted(s for it, s in zip(items, scores) if it["brief"] != b)
        thr = others[int(budget * len(others))]
        for i, it in enumerate(items):
            if it["brief"] == b:
                flagged[i] = scores[i] < thr
    n_bad = sum(bad)
    auto = [not f for f in flagged]
    return {"flag_rate": round(sum(flagged) / len(items), 3),
            "catch_recall": round(sum(f and x for f, x in zip(flagged, bad)) / n_bad, 3),
            "auto_pass_good_rate": round(sum(a and not x for a, x in zip(auto, bad)) / sum(auto), 3) if any(auto) else None}


def evaluate(items, results):
    bad = [it["cosine_to_gold"] < USABLE for it in items]
    gold = [it["cosine_to_gold"] for it in items]
    report = {"n_fields": len(items), "n_bad": sum(bad), "base_good_rate": round(1 - sum(bad) / len(items), 3),
              "review_budget": BUDGET, "checkers": {}}
    # 现在的路由：只有"通过 / 不通过"两档
    cur_flag = [it["status"] != "通过" for it in items]
    report["current_router"] = {"flag_rate": round(sum(cur_flag) / len(items), 3),
                                "catch_recall": round(sum(f and x for f, x in zip(cur_flag, bad)) / sum(bad), 3),
                                "auto_pass_good_rate": round(sum((not f) and not x for f, x in zip(cur_flag, bad)) / sum(not f for f in cur_flag), 3),
                                "auc": round(auc([0 if f else 1 for f in cur_flag], bad), 3)}
    report["random_baseline"] = {"catch_recall_at_budget": BUDGET, "auc": 0.5}
    names = ["current_overlap", "C_source_similarity", "A_self_consistency", "B_ai_judge"]
    for name in names:
        sc = [r[name] if r[name] is not None else -1 for r in results]    # 检查员失败 = 最可疑
        report["checkers"][name] = {
            "auc": round(auc(sc, bad), 3),
            "spearman_with_gold": round(float(spearmanr(sc, gold).statistic), 3),
            "catch_recall_at_budget": round(recall_at_budget(sc, bad), 3),
            "leave_one_brief_out": leave_one_brief_out(items, sc, bad),
            "failures": sum(r[name] is None for r in results)}
    judge = [r["B_ai_judge"] for r in results]
    report["judge_score_distribution"] = {s: judge.count(s) for s in sorted(set(j for j in judge if j is not None))}
    return report


def print_report(rep, spent):
    print(f"\n{rep['n_fields']} 个事实栏，不合格 {rep['n_bad']} 个（整体合格率 {rep['base_good_rate']}）。人工审核量上限 {int(rep['review_budget']*100)}%。花费 ${spent:.4f}")
    c = rep["current_router"]
    print(f"\n现在的路由：转人工 {c['flag_rate']:.0%}，拦下不合格 {c['catch_recall']:.0%}，自动通过的合格率 {c['auto_pass_good_rate']:.0%}，AUC {c['auc']}")
    print(f"瞎猜：转人工 30% 能拦下 30%，AUC 0.5\n")
    print(f"{'检查员':<22}{'AUC':>6}{'与参考答案相关':>12}{'30%预算拦下':>12}   留一法：{'转人工':>6}{'拦下':>6}{'自动通过合格率':>10}")
    for n, m in rep["checkers"].items():
        l = m["leave_one_brief_out"]
        print(f"{n:<22}{m['auc']:>6.3f}{m['spearman_with_gold']:>12.3f}{m['catch_recall_at_budget']:>12.0%}   "
              f"      {l['flag_rate']:>6.0%}{l['catch_recall']:>6.0%}{l['auto_pass_good_rate']:>10.0%}")
    print(f"\nAI 审核员打分分布：{rep['judge_score_distribution']}")


def main():
    REPORTS.mkdir(exist_ok=True)
    items = load_items()
    emb = EmbeddingClient(api_key=exp_cfg.OPENAI_API_KEY, model=exp_cfg.EMBEDDING_MODEL,
                          cache_path=REPORTS / "embedding_cache.jsonl")
    cache = REPORTS / "checkers_raw.json"                                    # 含客户内容：只留本地
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        results, spent = saved["results"], saved["spent"]
    else:
        results, spent = run_checkers(items, emb)
        emb.save_cache()
        cache.write_text(json.dumps({"results": results, "spent": spent}, ensure_ascii=False), encoding="utf-8")
    rep = evaluate(items, results)
    rep["spent_usd"] = round(spent, 4)
    (REPORTS / "checkers_summary.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print_report(rep, spent)


if __name__ == "__main__":
    sys.exit(main())
