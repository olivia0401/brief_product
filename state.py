# state.py：在图里流动的那一份数据
from typing import Annotated, TypedDict


def merge(a: dict, b: dict) -> dict:
    """reducer：9 栏并行写 items 时，把各自的结果合并进同一个字典。
    带 __reset__ 标记表示"清空重来"（客户补充简报后），和普通合并分开"""
    if b.get("__reset__"):
        return {}
    return {**a, **b}


class Item(TypedDict, total=False):
    value: str | list[str]       # 模型写的内容；关键词是列表
    quotes: list[str]            # 模型抄出的原话
    evidence: list[str]          # 程序核对后确认在简报里的原话（只证明"原话存在"，不证明"原话支持结论"）
    status: str                  # 栏状态：通过 / 建议 / 需人工确认 / 未知 / 格式不合格 / 调用失败
    problems: list[str]
    cost_usd: float
    latency_s: float
    attempts: int
    config_id: str


class State(TypedDict, total=False):
    client_id: str
    brief_id: str
    source_hash: str            # 当前输入文件的内容哈希（客户补充后变）
    brief_path: str
    reviewer: str
    recipe_version: str          # 任务启动时固定；恢复与客户补充时沿用，交付前核对未变
    recipe_hash: str             # 配置文件内容哈希，与版本名一起核对
    brief: dict                  # 6 个输入字段
    missing_fields: list[str]
    contradictions: list         # [(规则名, 甲组命中, 乙组命中, 受影响的栏)]
    items: Annotated[dict[str, Item], merge]
    clarification_round: int     # 已问客户几轮
    clarification_draft: str     # 为空表示不再问
    clarification_log: list      # 每轮：谁决定（rule / model）、动作、理由、问题、费用
    prior_cost_usd: float        # 客户补充前几轮的模型花费；不声明的话 LangGraph 会丢掉这个字段
    confirmed_at: str           # 人工确认时间，human_review 写入后不再变
    confirmed: dict              # 栏 → {value, action, evidence, evidence_source, edited_by_human}
    review_errors: list          # 核对回答或交付格式的问题；非空就回到 human_review 重新问
    delivered_path: str
