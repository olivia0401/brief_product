# demo_run.py：用公开示例简报（虚构的 Acme 智能家居）真实跑一次完整流程，不需要手动输入。
# 分三步、三个独立进程运行，模拟"隔几天再继续"：
#   python demo/demo_run.py start            简报进来 → 缺受众 → 停下等客户
#   python demo/demo_run.py client_replied   客户补充后 → 重新提取 9 栏 → 停下等经理核对
#   python demo/demo_run.py review           经理逐栏确认 → 交付
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from graph import build
import nodes

CLIENT = "demo-acme"
V1, V2 = HERE / "acme_v1.txt", HERE / "acme_v2_client_reply.txt"
root = nodes.data_dir(CLIENT); root.mkdir(parents=True, exist_ok=True)
brief_id = nodes.brief_id_for(V1)                      # 任务身份由第一份简报决定，客户补充后不变
config = {"configurable": {"thread_id": f"{CLIENT}:{brief_id}"}}


def show_pause(result):
    p = result["__interrupt__"][0].value
    if p["kind"] == "ask_client":
        print("\n⏸  流程暂停：等客户回复")
        print("   缺的关键字段：", p["missing"])
        print("   AI 写的澄清邮件草稿：\n")
        print("   " + p["draft"].replace("\n", "\n   "))
    else:
        print("\n⏸  流程暂停：等经理逐栏核对\n")
        for task, it in p["items"].items():
            print(f"[{task}]  状态：{it['status']}")
            print(f"   AI 草稿：{it.get('value')}")
            print(f"   原文依据：{it.get('evidence')}")
            if it.get("problems"):
                print(f"   提示：{it['problems']}")
            print(f"   费用：${it.get('cost_usd', 0):.5f}\n")
    return p


step = sys.argv[1] if len(sys.argv) > 1 else "start"
with SqliteSaver.from_conn_string(str(root / "checkpoints.db")) as saver:
    app = build(saver)
    snap = app.get_state(config)
    if step == "start":
        if snap.values:
            raise SystemExit("这份示例已经开始过了：删掉 data/demo-acme 再重来")
        print("▶ 第 1 天：客户提交简报\n")
        print(V1.read_text(encoding="utf-8"))
        result = app.invoke({"client_id": CLIENT, "brief_path": str(V1), "brief_id": brief_id, "reviewer": "demo-manager"}, config)
        show_pause(result)
    elif step == "client_replied":
        print("▶ 第 3 天：新开一个进程，从存档里找到上次停下的位置")
        print("   存档里停在：", [i.value["kind"] for i in snap.interrupts])
        print("   经理决定发出澄清邮件，客户补充了受众：\n")
        print(V2.read_text(encoding="utf-8"))
        result = app.invoke(Command(resume={"action": "resent", "brief_path": str(V2)}), config)
        show_pause(result)
    elif step == "review":
        print("▶ 经理核对：新开一个进程，从存档继续")
        payload = snap.interrupts[0].value
        # 经理对照原文逐栏判断：大多数原样采用；三栏修改（依据都是简报原话）
        edits = {
            "category_relevant": {  # 格式不合格：14 词，要求 15-20
                "value": "Voice-controlled smart home hubs that connect lights, locks and thermostats, a consumer electronics product in the smart home category.",
                "evidence": ["Consumer electronics, smart home devices", "voice-controlled smart home hubs that connect lights, locks and thermostats"]},
            "feature_relevant": {   # AI 加了 "security"，简报没说
                "value": "Voice-controlled smart home hubs that connect lights, locks and thermostats, with setup that takes under five minutes.",
                "evidence": ["voice-controlled smart home hubs that connect lights, locks and thermostats", "Sets up in under five minutes."]},
            "context_relevant": {   # AI 加了 "flexible mobility"，简报没说
                "value": "Mid-30s renters in urban apartments who want connected-home convenience without paying a monthly subscription bill.",
                "evidence": ["Mid-30s renters in urban apartments who want connected-home convenience without a subscription bill."]},
        }
        answers = {}
        for task, it in payload["items"].items():
            if task in edits:
                answers[task] = {"action": "edit", **edits[task]}
            else:
                answers[task] = {"action": "accept"} if it.get("value") else {"action": "unknown"}
        print("   经理的决定：", {t: a["action"] for t, a in answers.items()})
        result = app.invoke(Command(resume=answers), config)
        out = Path(result["delivered_path"])
        print("\n✅ 已交付：", out)
        print(out.read_text(encoding="utf-8"))
        log = json.loads((root / "run_log" / out.name).read_text(encoding="utf-8"))
        print("\n本次总费用（两轮提取 + 澄清邮件）：$", log["total_cost_usd"])
