# run.py：命令行入口。新简报就启动图；有停下的记录就从那个中断继续
import argparse
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from graph import build
import nodes

ap = argparse.ArgumentParser(description="品牌简报提取")
ap.add_argument("brief", help="简报文件：.docx / .txt / .md / .yml / .json")
ap.add_argument("--client", required=True, help="客户编号（字母、数字、- _）")
ap.add_argument("--reviewer", default="cli", help="核对人，写进交付文件")
a = ap.parse_args()

root = nodes.data_dir(a.client)                                        # 校验编号，得到该客户自己的目录
root.mkdir(parents=True, exist_ok=True)
brief_id = nodes.brief_id_for(a.brief)                                 # 任务身份：文件名 + 首次内容哈希，之后不变
config = {"configurable": {"thread_id": f"{a.client}:{brief_id}"}}      # 同名不同内容是两次任务


def answer(payload):
    """把中断的内容展示给人，返回要送回图的答案"""
    if payload["kind"] == "ask_client":
        print(f"\n第 {payload['round'] + 1} 轮澄清。关键缺口：", payload["missing"], "矛盾：", payload["contradictions"])
        print("澄清邮件草稿：\n", payload["draft"])
        if input("问不问由你定：发给客户并等待补充(r) / 不问，自己处理(s)：").strip() == "r":
            return {"action": "resent", "brief_path": input("客户补充后的简报文件路径：").strip()}
        return {"action": "skip"}
    for err in payload.get("errors", []):
        print("上次的回答有问题：", err)
    for e in payload.get("clarification_log", []):
        print(f"澄清第 {e['round'] + 1} 轮：{e['action']}  {e.get('reason', '')}")
    answers = {}
    for task, item in payload["items"].items():
        print(f"\n[{task}] {item['status']} {item.get('problems', '')}\n  模型：{item.get('value')}\n  依据：{item.get('evidence')}")
        has_value = bool(item.get("value"))
        ok_default = item["status"] == "通过" and has_value                   # "通过"的栏回车即采用（批量确认的命令行版）
        valid = ("a", "e", "u") if has_value else ("e", "u")                  # 没有内容的栏不能"采用"
        hint = ("回车=采用 / e=修改 / u=确认未知" if ok_default else
                "a=核实无误、原样采用 / e=修改 / u=确认未知（必须选）" if has_value else "e=修改 / u=确认未知（这栏没有内容，必须选）")
        choice = input(f"  {hint}：").strip().lower()
        while choice not in valid and not (choice == "" and ok_default):      # 非"通过"的栏不能回车跳过
            choice = input(f"  请选 {' / '.join(valid)}：").strip().lower()
        if choice in ("", "a"):
            answers[task] = {"action": "accept"}
        elif choice == "u":
            answers[task] = {"action": "unknown"}
        else:
            value = input("  修改后的内容：").strip()
            evidence = [q.strip() for q in input("  依据（简报原话，分号分隔；没有就留空）：").split(";") if q.strip()]
            answers[task] = {"action": "edit", "value": value, "evidence": evidence}   # 依据来源由 human_review 按原话是否在简报里判定
    return answers


with SqliteSaver.from_conn_string(str(root / "checkpoints.db")) as saver:     # 连接在整个运行期间保持打开
    # durability="sync"：每个执行阶段的存档写完才往下走（默认是后台异步写，崩溃时可能丢掉刚完成那一步）
    app = build(saver)
    snap = app.get_state(config)                                              # thread_id 只负责"找到记录"
    if snap.values.get("delivered_path"):
        raise SystemExit(f"这份简报已经交付：{snap.values['delivered_path']}")
    pending = getattr(snap, "interrupts", ()) or ()
    if pending:                                                               # 上次停在中断：读出问题，用 Command 续
        print("发现上次停下的记录，从中断处继续。")
        result = app.invoke(Command(resume=answer(pending[0].value)), config, durability="sync")
    elif snap.next:                                                           # 有存档但没停在中断上：上次中途崩溃
        print("发现上次中途退出的记录，从存档处继续。")                        # 传 None 才是续跑；重新给输入会从头重跑，旧状态还留着
        result = app.invoke(None, config, durability="sync")
    else:                                                                     # 新简报：提交输入
        result = app.invoke({"client_id": a.client, "brief_path": a.brief, "brief_id": brief_id, "reviewer": a.reviewer}, config, durability="sync")
    while "__interrupt__" in result:                                          # 同一进程里停了就问，答完继续
        result = app.invoke(Command(resume=answer(result["__interrupt__"][0].value)), config, durability="sync")
    print("\n已交付：", result["delivered_path"])
