# llm.py：调模型。LangChain 封装 + 结构化输出 + 按实验价格表算费用
import time
from pydantic import BaseModel, Field
from langchain.chat_models import init_chat_model

RECIPE = None   # 由 nodes.py 启动时注入


class Extraction(BaseModel):
    """一次调用同时返回内容和出处；澄清邮件草稿也用它，quotes 留空即可"""
    value: str = Field(description="The requested sentence or comma-separated keyword list, nothing else")
    quotes: list[str] = Field(description="Phrases copied word for word from the brief that support the value; empty if none")


QUOTE_SUFFIX = ("\n\nAlso return, as `quotes`, the phrases copied word for word from the brief "
                "that support your answer. If nothing in the brief supports it, return an empty list.")


def make_model(model_key, schema=Extraction):
    """按 recipe 里的模型编号建 LangChain 模型。参数显式写出；是否真与实验等价，要在上线步骤 1 核对"""
    spec = RECIPE["models"][model_key]
    params = {k: spec[k] for k in ("temperature", "reasoning_effort") if k in spec}   # 推理模型不接受 temperature，只传配置里写了的
    model = init_chat_model(spec["id"], model_provider=spec["provider"],   # max_retries=0：重试由 call() 自己做，每次尝试都记账
                            max_tokens=spec.get("max_tokens", 1000), timeout=60, max_retries=0, **params)
    return model.with_structured_output(schema, include_raw=True)


def cost_of(usage, model_key):
    p = RECIPE["models"][model_key]["price_per_1m"]
    return (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]) / 1e6


# 只有暂时性错误才重试（限流、超时、连接断、服务端 5xx）；密钥错、解析失败重试也没用
TRANSIENT = {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError", "TimeoutError", "ReadTimeout"}
MAX_RETRIES = 2
BACKOFF_S = (1.0, 3.0)
SLEEP = time.sleep                             # 测试里替换掉，不真的等


def call(model_key, prompt, schema=Extraction, suffix=QUOTE_SUFFIX):
    """返回 {parsed, usage, input_tokens, output_tokens, cost_usd, latency_s, retries, ok, error}；从不抛异常。
    include_raw=True 时返回 {raw, parsed, parsing_error}：解析失败不抛错，parsed 是 None——必须自己查。
    解析失败也花了 token，费用照记；暂时性错误重试时，每次尝试的 token 和费用累加，重试次数写进 retries"""
    t0 = time.time()
    tokens_in = tokens_out = 0
    cost, retries, usage = 0.0, 0, {}
    while True:
        try:
            out = make_model(model_key, schema).invoke(prompt + suffix)
            usage = getattr(out.get("raw"), "usage_metadata", None) or {}
            tokens_in += usage.get("input_tokens", 0); tokens_out += usage.get("output_tokens", 0)
            cost += cost_of(usage, model_key) if usage else 0.0
            if out.get("parsing_error") or out.get("parsed") is None:
                raise ValueError(f"结构化输出解析失败: {out.get('parsing_error')}")
            return {"parsed": out["parsed"], "usage": usage, "input_tokens": tokens_in, "output_tokens": tokens_out,
                    "cost_usd": cost, "latency_s": time.time() - t0, "retries": retries, "ok": True, "error": None}
        except Exception as e:                 # 超时、限流、密钥错、解析失败都落到这里
            if type(e).__name__ in TRANSIENT and retries < MAX_RETRIES:
                SLEEP(BACKOFF_S[retries]); retries += 1
                continue
            return {"parsed": None, "usage": usage, "input_tokens": tokens_in, "output_tokens": tokens_out,
                    "cost_usd": cost, "latency_s": time.time() - t0, "retries": retries, "ok": False,
                    "error": f"{type(e).__name__}: {e}"}
