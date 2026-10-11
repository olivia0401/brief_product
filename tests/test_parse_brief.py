# tests/test_parse_brief.py：读简报。列表字段要和实验的 prompt 渲染一致
import json
import nodes
from parse_brief import load_brief


def test_list_field_renders_like_the_experiment(tmp_path):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"product": "hubs", "personality": ["Approachable", "Witty"]}), encoding="utf-8")
    brief, _ = load_brief(p)
    assert brief["personality"] == "- Approachable\n- Witty"           # 不是 "['Approachable', 'Witty']"
    as_list = nodes.build_prompt("keywords", ["personality"], {"personality": ["Approachable", "Witty"]},
                                 prompt_version="B", templates=nodes.TEMPLATES)
    as_loaded = nodes.build_prompt("keywords", ["personality"], brief, prompt_version="B", templates=nodes.TEMPLATES)
    assert as_loaded == as_list                                          # 上线和实验喂给模型的文字一样
