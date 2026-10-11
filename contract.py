# contract.py：交给命名流程的数据格式（接口约定），以及写出前的检查
CONTRACT_VERSION = "key-info/v1"
KINDS = {"fact", "interpretation", "gap"}          # 事实栏 / 解读栏 / 缺口
EVIDENCE_SOURCES = {"brief", "human"}             # 原话在简报里 / 由人工补充。客户补充的信息进入新简报文件，仍算 brief


def build_output(brief_id, source_hash, client_id, recipe_version, confirmed_at, confirmed_by, items):
    """items：{任务: {"value", "kind", "evidence", "evidence_source", "status", "ai_status", "edited_by_human"}}"""
    return {"contract_version": CONTRACT_VERSION, "brief_id": brief_id, "source_hash": source_hash, "client_id": client_id,
            "recipe_version": recipe_version, "confirmed_at": confirmed_at, "confirmed_by": confirmed_by, "items": items}


def validate(doc, tasks, specs):
    """返回问题列表；空列表表示可以交给命名流程。specs 是 recipe 的 tasks，用来查字数和关键词个数"""
    problems = []
    if doc.get("contract_version") != CONTRACT_VERSION:
        problems.append("contract_version 不对")
    for key in ("brief_id", "source_hash", "client_id", "recipe_version", "confirmed_at", "confirmed_by"):
        if not doc.get(key):
            problems.append(f"缺少 {key}")
    items = doc.get("items", {})
    for t in tasks:
        it = items.get(t)
        if it is None:
            problems.append(f"缺少任务 {t}"); continue
        v = it.get("value")
        if it.get("status") != "已确认":
            problems.append(f"{t} 还没有人工确认")
        if not isinstance(it.get("evidence"), list) or it.get("evidence_source") not in EVIDENCE_SOURCES:
            problems.append(f"{t} 的 evidence 要是列表，evidence_source 要是 {sorted(EVIDENCE_SOURCES)} 之一")
        if it.get("kind") not in KINDS or (it.get("kind") == "gap") != (v == "未知"):
            problems.append(f"{t} 的 kind 要是 {sorted(KINDS)} 之一，且'未知'必须是 gap")
        if v == "未知":                                               # 任何栏都允许"未知"交付：由人确认过，下游自行处理
            continue
        if t == "keywords":
            if not isinstance(v, list) or not all(isinstance(k, str) and k.strip() for k in v):
                problems.append("keywords 必须是非空字符串列表")
            elif len(v) != specs[t]["keyword_count"]:
                problems.append(f"keywords 要 {specs[t]['keyword_count']} 个，现有 {len(v)}")
        elif not v:
            problems.append(f"{t} 没有内容")
        else:
            n = len(str(v).split())
            if not (specs[t]["min_words"] <= n <= specs[t]["max_words"]):
                problems.append(f"{t} 长度 {n} 词，要求 {specs[t]['min_words']}-{specs[t]['max_words']}")
    return problems
