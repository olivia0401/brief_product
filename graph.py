# graph.py：把节点连成图，接检查点
from langgraph.graph import StateGraph, START, END

from state import State
import nodes as n


def build(checkpointer):
    """checkpointer 由入口创建并管理生命周期（见 run.py）：
    SqliteSaver.from_conn_string 是上下文管理器，数据库连接要在整个运行和恢复期间保持打开"""
    g = StateGraph(State)
    g.add_node("read_brief", n.read_brief)
    g.add_node("find_conflicts", n.find_conflicts)
    g.add_node("extract_one", n.extract_one)
    g.add_node("draft_clarification", n.draft_clarification)
    g.add_node("ask_client", n.ask_client)
    g.add_node("human_review", n.human_review)
    g.add_node("deliver", n.deliver)

    g.add_edge(START, "read_brief")
    g.add_edge("read_brief", "find_conflicts")
    g.add_conditional_edges("find_conflicts", n.fan_out, ["extract_one"])       # 9 栏并行（Send）
    g.add_conditional_edges("extract_one", n.needs_clarification,                   # 只有关键缺口或矛盾才问
                            {"draft_clarification": "draft_clarification", "human_review": "human_review"})
    g.add_conditional_edges("draft_clarification", n.after_draft,
                            {"ask_client": "ask_client", "human_review": "human_review"})       # 有草稿 / 满轮数
    g.add_conditional_edges("ask_client", n.after_ask,
                            {"read_brief": "read_brief", "human_review": "human_review"})       # 补充后重跑 / 跳过
    g.add_conditional_edges("human_review", n.after_review, ["human_review", "deliver"])   # 回答不完整就重新问
    g.add_conditional_edges("deliver", n.after_deliver, {"human_review": "human_review", "__end__": END})   # 格式不合格退回核对

    return g.compile(checkpointer=checkpointer)
