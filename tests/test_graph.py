# tests/test_graph.py：用假模型跑整张图，不花钱。每次 invoke 都重开数据库、重建图，模拟"关掉程序再运行"
import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
import llm, nodes
from graph import build

BRIEF = ("Product: voice-controlled smart home hubs that connect lights, locks and thermostats\n"
         "Audience: renters in urban apartments\nBrand strategy: Friendly tech that respects your time\n"
         "Differentiators: no subscription\nBusiness category: smart home\nPersonality: witty")
FAKE = {   # 栏 → 假的模型回复。引用必须真的在 BRIEF 里，否则会被核对拦下
    "function_relevant": llm.Extraction(value="A voice-controlled smart home hub that connects lights, locks and thermostats and sets up in a few minutes.",
                                        quotes=["voice-controlled smart home hubs that connect lights, locks and thermostats"]),
    "emotion_relevant": llm.Extraction(value="Calm confidence that the home is quietly taken care of, with time left over for living well.",
                                       quotes=["Friendly tech that respects your time"]),
}
GENERIC = llm.Extraction(value="A generic sentence that is long enough to pass the format check for this task right now.", quotes=[])


def fake_call(model_key, prompt, schema=llm.Extraction, suffix=""):
    parsed = next((p for t, p in FAKE.items() if t.split("_")[0] in prompt.lower()), GENERIC)
    return {"parsed": parsed, "usage": {}, "cost_usd": 0.0002, "latency_s": 0.01, "ok": True, "error": None}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(llm, "call", fake_call)
    monkeypatch.setattr(nodes, "HERE", tmp_path)                # 记录写到临时目录
    brief = tmp_path / "b.txt"
    brief.write_text(BRIEF, encoding="utf-8")
    return tmp_path, brief


def run(tmp_path, cfg, inp):
    with SqliteSaver.from_conn_string(str(tmp_path / "ck.db")) as saver:   # 新连接、新图
        return build(saver).invoke(inp, cfg, durability="sync")         # 和 run.py 一样同步写存档


def test_resume_across_processes_and_deliver(env):
    tmp_path, brief = env
    cfg = {"configurable": {"thread_id": "t1"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief)})
    p = r["__interrupt__"][0].value                                # 停在人工核对
    assert p["kind"] == "human_review"
    assert p["items"]["function_relevant"]["status"] == "通过"      # 事实栏，引用在简报里，用词重合
    assert p["items"]["emotion_relevant"]["status"] == "建议"        # 解读栏：标"建议"，转人工判断
    assert p["items"]["concept_relevant"]["status"] == "需人工确认"  # 没给引用
    answers = {t: {"action": "accept"} for t in p["items"]}        # 人核实后原样采用，不必改字
    answers["keywords"] = {"action": "unknown"}                      # 关键词也可以"确认未知"后交付
    r = run(tmp_path, cfg, Command(resume=answers))                 # 关掉再开：从中断处续，提取节点不重跑
    import json
    doc = json.loads(open(r["delivered_path"], encoding="utf-8").read())
    assert doc["items"]["emotion_relevant"]["kind"] == "interpretation" and doc["items"]["function_relevant"]["kind"] == "fact"
    assert doc["items"]["keywords"] == {**doc["items"]["keywords"], "value": "未知", "kind": "gap"}
    assert (tmp_path / "data" / "acme" / "edits_for_eval" / f"{nodes.brief_id_for(brief)}.json").exists()


def test_edit_keeps_evidence_source_and_freezes_confirmed_at(env):
    tmp_path, brief = env
    cfg = {"configurable": {"thread_id": "t5"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief), "brief_id": nodes.brief_id_for(brief)})
    p = r["__interrupt__"][0].value
    answers = {t: {"action": "accept"} for t in p["items"]}
    answers["function_relevant"] = {"action": "edit", "value": "A voice-controlled smart home hub for lights, locks and thermostats that renters can set up in minutes.",
                                    "evidence": ["voice-controlled smart home hubs that connect lights, locks and thermostats"]}   # 改了措辞，依据仍是简报原话
    answers["keywords"] = {"action": "unknown"}
    r = run(tmp_path, cfg, Command(resume=answers))
    import json
    doc = json.loads(open(r["delivered_path"], encoding="utf-8").read())
    it = doc["items"]["function_relevant"]
    assert it["edited_by_human"] is True and it["evidence_source"] == "brief"   # 谁改的 ≠ 依据来自哪里
    assert doc["source_hash"] == nodes.source_hash_for(brief) and doc["confirmed_at"] == r["confirmed_at"]
    assert doc["brief_id"] == nodes.brief_id_for(brief)


def test_same_name_different_content_is_new_run(env):
    tmp_path, brief = env
    a = nodes.brief_id_for(brief)
    brief.write_text(BRIEF + "\nextra line", encoding="utf-8")
    assert nodes.brief_id_for(brief) != a                            # 同名文件内容变了，是另一次任务


def test_missing_field_asks_client_first(env):
    tmp_path, brief = env
    brief.write_text("Product: smart hubs\nBrand strategy: friendly\n", encoding="utf-8")   # 缺 audience 等
    cfg = {"configurable": {"thread_id": "t2"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief)})
    p = r["__interrupt__"][0].value
    assert p["kind"] == "ask_client" and "audience" in p["missing"]  # 先停在"问客户"
    r = run(tmp_path, cfg, Command(resume={"action": "skip"}))
    assert r["__interrupt__"][0].value["items"]["context_relevant"]["status"] == "未知"


def test_only_critical_gaps_ask_client(env):
    tmp_path, brief = env
    brief.write_text("Product: smart hubs\nAudience: renters\nDifferentiators: cheap\nBusiness category: smart home\n", encoding="utf-8")
    cfg = {"configurable": {"thread_id": "t4"}}                        # 缺 brand_strategy、personality：不算关键，不问
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief)})
    p = r["__interrupt__"][0].value
    assert p["kind"] == "human_review"
    assert p["items"]["concept_relevant"]["status"] == "未知"          # 缺口保留未知，由人决定要不要补


def test_clarification_stops_at_cap(env):
    tmp_path, brief = env
    brief.write_text("Product: smart hubs\n", encoding="utf-8")        # 缺 audience：关键缺口
    cfg = {"configurable": {"thread_id": "t3"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief)})
    assert r["__interrupt__"][0].value["round"] == 0
    r = run(tmp_path, cfg, Command(resume={"action": "resent", "brief_path": str(brief)}))   # 客户回了，但什么都没补
    p = r["__interrupt__"][0].value
    assert p["kind"] == "human_review"                               # 第一版只问一轮：不再问，转人工确认
    assert [e["action"] for e in p["clarification_log"]] == ["draft", "stop"]


def test_resent_keeps_prior_cost(env):
    tmp_path, brief = env
    brief.write_text("Product: smart hubs\n", encoding="utf-8")        # 缺 audience：先问客户
    cfg = {"configurable": {"thread_id": "t6"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief)})
    r = run(tmp_path, cfg, Command(resume={"action": "resent", "brief_path": str(brief)}))   # 补充后重跑：第一轮的花费要累计
    p = r["__interrupt__"][0].value
    r = run(tmp_path, cfg, Command(resume={t: {"action": "unknown"} for t in p["items"]}))
    import json
    log = json.loads((tmp_path / "data" / "acme" / "run_log" / f"{nodes.brief_id_for(brief)}.json").read_text(encoding="utf-8"))
    this_round = sum(i["cost_usd"] for i in p["items"].values()) + sum(e.get("cost_usd", 0) for e in p["clarification_log"])
    assert this_round > 0 and log["total_cost_usd"] > round(this_round, 6)   # 第一轮提取的花费没有丢


def test_parse_failure_is_not_ok(monkeypatch):
    class Broken:
        def invoke(self, prompt):
            return {"raw": None, "parsed": None, "parsing_error": ValueError("bad json")}
    monkeypatch.setattr(llm, "make_model", lambda *a, **k: Broken())
    r = llm.call("gpt5mini", "x")
    assert r["ok"] is False and "解析失败" in r["error"]


def test_bad_client_id_rejected():
    with pytest.raises(ValueError):
        nodes.data_dir("../other")


def test_crash_after_delivery_write_resumes_without_rerun(env, monkeypatch):
    """交付文件写完、还没存档就崩：续跑（invoke(None)）只重跑交付，不重调模型、不重新核对，确认时间不变"""
    tmp_path, brief = env
    cfg = {"configurable": {"thread_id": "t9"}}
    calls = []
    monkeypatch.setattr(llm, "call", lambda *a, **k: calls.append(1) or fake_call(*a, **k))
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief), "brief_id": nodes.brief_id_for(brief)})
    answers = {t: {"action": "accept"} for t in r["__interrupt__"][0].value["items"]}
    answers["keywords"] = {"action": "unknown"}
    real_deliver = nodes.deliver

    def crash_after_write(state):
        real_deliver(state)                                            # 文件已写好
        raise RuntimeError("进程在存档前崩溃")
    monkeypatch.setattr(nodes, "deliver", crash_after_write)
    with pytest.raises(RuntimeError):
        run(tmp_path, cfg, Command(resume=answers))
    n_calls, out = len(calls), tmp_path / "data" / "acme" / "confirmed" / f"{nodes.brief_id_for(brief)}.json"
    import json
    first = json.loads(out.read_text(encoding="utf-8"))
    monkeypatch.setattr(nodes, "deliver", real_deliver)
    r = run(tmp_path, cfg, None)                                        # run.py 在"有存档、没中断"时就这样续跑
    second = json.loads(open(r["delivered_path"], encoding="utf-8").read())
    assert len(calls) == n_calls                                         # 没有重新调用模型
    assert second["confirmed_at"] == first["confirmed_at"] and second == first   # 同一份交付，覆盖写结果不变


def test_bad_review_answer_asks_again(env):
    """核对回答不完整时不抛错，而是带着问题重新停在核对；改正后照常交付（抛错会让坏回答留在存档里反复重放）"""
    tmp_path, brief = env
    cfg = {"configurable": {"thread_id": "t10"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief), "brief_id": nodes.brief_id_for(brief)})
    items = r["__interrupt__"][0].value["items"]
    assert items["concept_relevant"]["status"] == "需人工确认"
    bad = {t: {"action": "accept"} for t in items if t != "concept_relevant"}   # 漏答一栏非"通过"的栏
    bad["keywords"] = {"action": "unknown"}
    r = run(tmp_path, cfg, Command(resume=bad))
    p = r["__interrupt__"][0].value
    assert p["kind"] == "human_review" and any("concept_relevant" in e for e in p["errors"])
    good = {**bad, "concept_relevant": {"action": "unknown"}}
    r = run(tmp_path, cfg, Command(resume=good))
    assert r["delivered_path"] and r["review_errors"] == []


def test_invalid_delivery_goes_back_to_review(env):
    """人把关键词改成不足 10 个：交付检查不通过时不抛错，退回核对并说明原因；改正后交付"""
    tmp_path, brief = env
    cfg = {"configurable": {"thread_id": "t11"}}
    r = run(tmp_path, cfg, {"client_id": "acme", "brief_path": str(brief), "brief_id": nodes.brief_id_for(brief)})
    items = r["__interrupt__"][0].value["items"]
    answers = {t: {"action": "accept"} if items[t].get("value") else {"action": "unknown"} for t in items}
    answers["concept_relevant"] = {"action": "unknown"}
    answers["keywords"] = {"action": "edit", "value": "renters, hubs", "evidence": []}   # 只有 2 个词，交付格式要求 10 个
    r = run(tmp_path, cfg, Command(resume=answers))
    p = r["__interrupt__"][0].value
    assert p["kind"] == "human_review" and any("交付格式不合格" in e for e in p["errors"])
    answers["keywords"] = {"action": "unknown"}
    r = run(tmp_path, cfg, Command(resume=answers))
    assert r["delivered_path"] and r["review_errors"] == []
