# parse_brief.py：把客户上传的简报读成六个字段
# 支持 .yml/.yaml/.json（字段名直接对应），.docx，以及 .txt/.md（每段以"字段名："开头）
import json
import re
from pathlib import Path

import yaml

FIELDS = ["business_category", "product", "differentiators", "audience", "brand_strategy", "personality"]

# 文字简报里可能出现的字段标题 → 统一的字段名
ALIASES = {
    "business_category": ["business category", "category", "industry", "品类", "行业"],
    "product": ["product", "service", "product or service", "产品", "服务"],
    "differentiators": ["differentiators", "differentiation", "usp", "差异点", "优势"],
    "audience": ["audience", "target audience", "customers", "受众", "目标客户"],
    "brand_strategy": ["brand strategy", "strategy", "positioning", "品牌策略", "定位"],
    "personality": ["personality", "brand personality", "tone", "个性", "品牌个性"],
}


def _field_for(label):
    """把一个标题（例如 'Target audience'）对应到字段名；对不上返回 None"""
    label = label.strip().lower()
    for field, names in ALIASES.items():
        if label in names:
            return field
    return None


def parse_text(text):
    """文字简报：遇到'标题：'或'标题:'开头的行，就开始一个新字段"""
    brief, current = {}, None
    for line in text.splitlines():
        m = re.match(r"^\s*#*\s*([^:：]{1,40})[:：]\s*(.*)$", line)
        field = _field_for(m.group(1)) if m else None
        if field:
            current = field
            brief[field] = m.group(2).strip()
        elif current and line.strip():
            brief[current] = (brief[current] + " " + line.strip()).strip()   # 续行
    return brief


def load_brief(path):
    """按扩展名读取简报，返回 (字段字典, 缺失的字段列表)"""
    path = Path(path)
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8") if suffix != ".docx" else ""
    if suffix in (".yml", ".yaml"):
        data = yaml.safe_load(raw)
        brief = data[0] if isinstance(data, list) else data   # 实验文件是一串简报，产品一次只处理一份
    elif suffix == ".json":
        brief = json.loads(raw)
    elif suffix == ".docx":
        import docx                                            # pip install python-docx
        d = docx.Document(path)
        lines = [p.text for p in d.paragraphs]
        for table in d.tables:                                 # 常见模板是两列表格：标题 | 内容
            for row in table.rows:
                cells = [c.text.strip() for c in row.cells]
                if len(cells) >= 2:
                    lines.append(f"{cells[0]}: {' '.join(cells[1:])}")
        brief = parse_text("\n".join(lines))
    elif suffix in (".txt", ".md"):
        brief = parse_text(raw)
    else:
        raise ValueError(f"暂不支持 {suffix}：请先把 PDF 另存为 .docx 或 .txt，再上传")
    if not isinstance(brief, dict) or not brief:
        raise ValueError("简报为空或格式不对：需要 '字段名: 内容' 的键值")
    brief = {(_field_for(k) or k): (str(v).strip() if v is not None else "") for k, v in brief.items()}   # YAML/JSON 的键也认别名
    missing = [f for f in FIELDS if not brief.get(f)]
    return brief, missing
