# experiment_repo.py：找到实验仓库（prompt-evaluation）。提示词和关键词解析直接复用实验代码，保证上线和实验用的是同一份
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
CANDIDATES = [HERE.parent / "prompt-evaluation", HERE.parent / "prompt test"]   # 和本仓库并排克隆时的常见目录名


def locate():
    """优先用环境变量 PROMPT_EVAL_DIR；否则在同级目录里找。找不到就说清楚怎么设，不静默失败"""
    env = os.environ.get("PROMPT_EVAL_DIR")
    paths = [Path(env)] if env else CANDIDATES
    for p in paths:
        if (p / "src" / "prompt_builder.py").exists() and (p / "prompts.txt").exists():
            return p.resolve()
    raise RuntimeError("找不到实验仓库 prompt-evaluation：请把它克隆到本仓库同级目录，"
                       f"或设置环境变量 PROMPT_EVAL_DIR。已查找：{[str(p) for p in paths]}")
