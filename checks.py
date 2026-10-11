# checks.py：输出检查。格式对不对、有没有原文依据、简报里有没有矛盾
import re

STOPWORDS = set("a an the and or of to for in on with by at from that this is are be as it its your our their "
                "who what which you we they into more most very".split())


def check_sentence(text, spec):
    """一句话任务：长度在范围内；不以引号或开场白开头"""
    problems = []
    n = len(text.split())
    if not (spec["min_words"] <= n <= spec["max_words"]):
        problems.append(f"长度 {n} 词，要求 {spec['min_words']}-{spec['max_words']}")
    if text.startswith(('"', "“", "'")) or text.lower().startswith(("here is", "summary:")):
        problems.append("带了引号或开场白")
    return problems


def normalise(text):
    """比较前统一格式：小写、去掉标点、合并空白"""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def lexical_overlap(output, source):
    """输出里的实词，有多少比例在原文里出现过（0 到 1）"""
    words = [w for w in normalise(output).split() if w not in STOPWORDS and len(w) > 2]
    if not words:
        return 0.0
    src = set(normalise(source).split())
    return sum(1 for w in words if w in src) / len(words)


def verify_quotes(quotes, source):
    """模型给出的每一段依据，都必须能在原文里逐字找到（忽略大小写、标点和空白）"""
    src = normalise(source)
    found = [q for q in quotes if q.strip() and normalise(q) in src]
    missing = [q for q in quotes if q.strip() and normalise(q) not in src]
    return found, missing


def find_contradictions(brief, rules):
    """简报里同时出现两组互相矛盾的说法时，返回 [(规则名, 甲方出现的词, 乙方出现的词, 受影响的任务)]"""
    text = " ".join(brief.values()).lower()
    hits = []
    for r in rules:
        a = [t for t in r["a"] if t.lower() in text]
        b = [t for t in r["b"] if t.lower() in text]
        if a and b:
            hits.append((r["name"], a, b, r["tasks"]))
    return hits


def unknown_fields(brief, fields):
    """这个任务需要的字段里，简报没有提供的"""
    return [f for f in fields if not brief.get(f)]
