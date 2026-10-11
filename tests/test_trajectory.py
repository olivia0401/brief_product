# tests/test_trajectory.py：轨迹评估的每个场景都是一条测试（场景定义在 evals/trajectory.py）
import pytest
from evals import trajectory as T


@pytest.mark.parametrize("sc", T.SCENARIOS, ids=[s.name for s in T.SCENARIOS])
def test_scenario(sc, tmp_path):
    r = T.run_scenario(sc, tmp_path)
    assert r["passed"], "\n".join(r["failures"])


def test_harness_catches_a_broken_guardrail(tmp_path, monkeypatch):
    """评估本身要能发现问题：关掉引用核对，编造引用的场景必须失败"""
    monkeypatch.setattr(T.nodes, "verify_quotes", lambda q, s: ([x for x in q if x.strip()], []))
    sc = next(s for s in T.SCENARIOS if s.name == "fabricated_quote")
    assert not T.run_scenario(sc, tmp_path)["passed"]
