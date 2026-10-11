# tests/test_llm.py：模型调用的重试与记账。用假模型，不花钱、不真的等
import pytest
import llm
import nodes   # noqa: F401  导入时把 recipe 注入 llm.RECIPE


class RateLimitError(Exception):        # 类名和 openai 的一致：call() 按类名判断是否暂时性错误
    pass


class AuthenticationError(Exception):
    pass


class Raw:
    def __init__(self, i, o):
        self.usage_metadata = {"input_tokens": i, "output_tokens": o}


def scripted(*steps):
    """按顺序执行：异常类就抛，dict 就作为模型返回"""
    steps = list(steps)

    class Model:
        def invoke(self, prompt):
            s = steps.pop(0)
            if isinstance(s, type):
                raise s("boom")
            return s
    return lambda *a, **k: Model()


OK = {"raw": Raw(100, 20), "parsed": llm.Extraction(value="v", quotes=[]), "parsing_error": None}


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    waits = []
    monkeypatch.setattr(llm, "SLEEP", waits.append)
    return waits


def test_transient_errors_are_retried_and_counted(monkeypatch, no_sleep):
    monkeypatch.setattr(llm, "make_model", scripted(RateLimitError, RateLimitError, OK))
    r = llm.call("gpt5mini", "x")
    assert r["ok"] and r["retries"] == 2 and no_sleep == list(llm.BACKOFF_S)
    assert (r["input_tokens"], r["output_tokens"]) == (100, 20)


def test_gives_up_after_max_retries(monkeypatch):
    monkeypatch.setattr(llm, "make_model", scripted(*[RateLimitError] * (llm.MAX_RETRIES + 1)))
    r = llm.call("gpt5mini", "x")
    assert not r["ok"] and r["retries"] == llm.MAX_RETRIES and "RateLimitError" in r["error"]


def test_auth_error_is_not_retried(monkeypatch):
    monkeypatch.setattr(llm, "make_model", scripted(AuthenticationError))
    r = llm.call("gpt5mini", "x")
    assert not r["ok"] and r["retries"] == 0


def test_parse_failure_still_bills_tokens(monkeypatch):
    bad = {"raw": Raw(80, 30), "parsed": None, "parsing_error": ValueError("bad json")}
    monkeypatch.setattr(llm, "make_model", scripted(bad))
    r = llm.call("gpt5mini", "x")
    assert not r["ok"] and (r["input_tokens"], r["output_tokens"]) == (80, 30) and r["cost_usd"] > 0
