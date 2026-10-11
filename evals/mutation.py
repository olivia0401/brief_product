# evals/mutation.py：检验评估本身。故意关掉 agent 的一处防护，再跑一遍轨迹评估：
# 至少一个场景失败 = 评估抓到了；全部通过 = 评估没测到这处防护。一个永远通过的评估什么也没测。
#
#   python -m evals.mutation               # 打印结果，写 evals/reports/mutation.json；有漏掉的就退出码 1
import json, sys, tempfile
from pathlib import Path

import nodes
from evals import trajectory as T

# (说明, nodes 里被换掉的名字, 换成什么)
MUTANTS = [
    ("disable quote verification", "verify_quotes", lambda q, s: ([x for x in q if x.strip()], [])),
    ("never ask the client", "needs_clarification", lambda st: "human_review"),
    ("remove the per-field budget", "PER_TASK_BUDGET_USD", 1e9),
    ("ignore contradictions in the brief", "find_conflicts", lambda st: {"contradictions": []}),
    ("skip the word-overlap check", "lexical_overlap", lambda a, b: 1.0),
]


def run_all():
    results = []
    for desc, attr, value in MUTANTS:
        original = getattr(nodes, attr)
        setattr(nodes, attr, value)
        try:
            with tempfile.TemporaryDirectory() as d:
                rep = T.run_all(Path(d))
        finally:
            setattr(nodes, attr, original)
        failed = [r["scenario"] for r in rep["results"] if not r["passed"]]
        results.append({"mutant": desc, "replaced": attr, "caught": bool(failed), "failed_scenarios": failed})
    return {"mutants": len(results), "caught": sum(r["caught"] for r in results), "results": results}


if __name__ == "__main__":
    report = run_all()
    out = Path(__file__).resolve().parent / "reports" / "mutation.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in report["results"]:
        print(f"{'CAUGHT' if r['caught'] else 'MISSED'}  {r['mutant']:<36} failing: {', '.join(r['failed_scenarios']) or '-'}")
    print(f"\n{report['caught']}/{report['mutants']} broken guardrails caught. Report: {out}")
    sys.exit(0 if report["caught"] == report["mutants"] else 1)
