"""Render a terminal-style replay of the real evaluation runs (evals/*.py).

Every number on screen is read from the eval reports in evals/reports/ (trajectory.json, mutation.json,
parity_summary*.json, checkers_summary.json). Only aggregate numbers are shown: client briefs stay private.
Grey italic lines marked '·' are narration, not program output.

    python demo/make_eval_replay.py demo/eval_demo.mp4
"""
import json, sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
REPORTS = Path(__file__).resolve().parent.parent / "evals" / "reports"
load = lambda name: json.loads((REPORTS / name).read_text(encoding="utf-8"))
traj, mut = load("trajectory.json"), load("mutation.json")
par, par_before, chk = load("parity_summary.json"), load("parity_summary_before_kwfix.json"), load("checkers_summary.json")
TOL = par["tolerance"]

W, H, FPS = 1280, 720, 15
F = "C:/Windows/Fonts/"
mono = ImageFont.truetype(F + "consola.ttf", 19)
mono_b = ImageFont.truetype(F + "consolab.ttf", 19)
mono_i = ImageFont.truetype(F + "consolai.ttf", 18)
ui = ImageFont.truetype(F + "segoeui.ttf", 15)
ui_b = ImageFont.truetype(F + "seguisb.ttf", 15)
big = ImageFont.truetype(F + "seguisb.ttf", 54)
mid = ImageFont.truetype(F + "segoeui.ttf", 26)
small = ImageFont.truetype(F + "segoeui.ttf", 18)

BG, BAR, TXT, DIM = "#1B2128", "#2B323B", "#DCE3EA", "#8A96A3"
TEAL, LIME, RED, AMBER, SLATE, WHITE = "#4ECDC4", "#C7F464", "#FF7B72", "#E3B341", "#454F5B", "#FFFFFF"
STEPS = ["Scenarios", "Break it on purpose", "23 real briefs", "Can 'pass' be trusted?", "Fix & re-measure"]
FOOTER = "Replay of real eval runs · client briefs stay private · only aggregate numbers are shown"

frames = []
def hold(img, secs):
    a = np.asarray(img.convert("RGB"))
    frames.extend([a] * max(1, int(secs * FPS)))


def card(kicker, title, sub, secs=3.4):
    im = Image.new("RGB", (W, H), SLATE); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=LIME)
    d.text((110, 220), kicker, font=ui_b, fill=LIME)
    d.text((106, 250), title, font=big, fill=WHITE)
    y = 340
    for line in sub.split("\n"):
        d.text((110, y), line, font=mid, fill="#DDE3E8"); y += 38
    d.rectangle([110, 200, 200, 205], fill=TEAL)
    d.text((110, H - 60), FOOTER, font=small, fill="#B9C2CB")
    hold(im, secs)


TOP, LH, LEFT, MAXW = 132, 27, 40, W - 80
term = []

def wrap(segs, indent=0):
    lines, cur, x = [], [], indent
    for text, font, colour in segs:
        for word in text.split(" "):
            token = word + " "
            w = font.getlength(token)
            if x + w > MAXW and cur:
                lines.append(cur); cur, x = [], indent + 24
            cur.append((token, font, colour)); x += w
    if cur: lines.append(cur)
    return [(indent if i == 0 else indent + 24, l) for i, l in enumerate(lines)]


def render(step_idx, done):
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 44], fill=BAR)
    for k, c in enumerate(["#FF5F57", "#FEBC2E", "#28C840"]):
        d.ellipse([18 + k * 22, 15, 32 + k * 22, 29], fill=c)
    d.text((100, 12), "brief_product — evals/   (replay of real eval runs)", font=ui, fill="#C9D1D9")
    x = 40
    for k, s in enumerate(STEPS):
        w = ui_b.getlength(s) + 28
        if k == step_idx:
            d.rounded_rectangle([x, 62, x + w, 96], 17, fill=TEAL); d.text((x + 14, 69), s, font=ui_b, fill="#10302E")
        elif k in done:
            d.rounded_rectangle([x, 62, x + w, 96], 17, outline=TEAL, width=2); d.text((x + 14, 69), s, font=ui_b, fill=TEAL)
        else:
            d.rounded_rectangle([x, 62, x + w, 96], 17, outline="#3A434D", width=2); d.text((x + 14, 69), s, font=ui, fill=DIM)
        x += w + 12
        if k < len(STEPS) - 1:
            d.text((x - 9, 68), "›", font=ui_b, fill=DIM)
    d.line([0, 112, W, 112], fill="#2E363F", width=1)
    y = TOP
    for indent, segs in term[-int((H - TOP - 40) / LH):]:
        xx = LEFT + indent
        for text, font, colour in segs:
            d.text((xx, y), text, font=font, fill=colour); xx += font.getlength(text)
        y += LH
    d.text((40, H - 30), "Console output condensed and shown in English. Grey '·' lines are narration, not program output.", font=ui, fill="#6E7A86")
    return im


state = {"step": 0, "done": set()}
def emit(segs, secs=0.45, indent=0):
    for line in wrap(segs, indent):
        term.append(line)
        hold(render(state["step"], state["done"]), secs)
def blank(): term.append((0, [("", mono, TXT)]))
def step(i): state["done"] |= set(range(i)); state["step"] = i
def cmd(c): emit([("$ ", mono_b, LIME), (c, mono, TXT)], 0.9)
def head(t): emit([("» ", mono_b, TEAL), (t, mono_b, TXT)], 0.8)
def note(t, secs=1.6): emit([("· " + t, mono_i, DIM)], secs, 18)
def stay(secs): hold(render(state["step"], state["done"]), secs)

SCENARIO_EN = {
    "happy_path": "complete brief, good answers: facts pass, interpretations become suggestions",
    "comma_separated_keywords": "keywords returned as one comma-separated line: still parsed",
    "fabricated_quote": "model invents a quote: field goes to a human",
    "no_evidence": "model gives no quote: field cannot pass",
    "inferred_content": "real quote, invented content: field goes to a human",
    "invented_keyword": "one keyword is not in the brief: flagged by name",
    "format_retry_recovers": "too short, fixed on retry: passes with one extra call",
    "format_fails_twice": "too short twice: marked as a format failure, no endless retries",
    "llm_error_isolated": "one API call fails: only that field is affected",
    "budget_stops_retry": "field budget used up: stops retrying",
    "contradiction_routes_to_human": "brief says premium AND cheapest: question drafted, fields flagged",
    "noncritical_gap_no_call": "non-critical field missing: no client email, no model call",
    "critical_gap_asks_client": "audience missing: pause and ask the client",
    "clarification_capped": "client replies without the info: ask only once",
    "incomplete_review_asks_again": "reviewer skips a field: ask again, no crash",
}
TASK_EN = {"concept_relevant": "concept", "position_relevant": "position", "emotion_relevant": "emotion",
           "function_relevant": "function", "benefit_relevant": "benefit", "category_relevant": "category",
           "feature_relevant": "feature", "context_relevant": "context", "keywords": "keywords (F1)"}
CHECKER_EN = {"current_overlap": "word overlap as a score", "A_self_consistency": "self-consistency (2 extra answers)",
              "C_source_similarity": "meaning similarity to the brief", "B_ai_judge": "AI judge (GPT-5-mini, 1-5)"}


def verdict(lo, hi):
    if hi < -TOL: return "worse", RED
    if lo < -TOL: return "can't rule out worse", AMBER
    if hi <= TOL: return "same", TEAL
    return "no worse", TEAL


pct = lambda x: f"{round(100 * x)}%"

# ================== the story ==================
card("EVALUATING THE AGENT", "Is it right? Can it be trusted?",
     "The same LangGraph brief-extraction agent, now measured:\nscripted scenarios, 23 real client briefs, and its own 'pass' label.", 4.2)

# ---- 1 trajectory ----
card("STEP 1 · FREE, RUNS IN CI", "Does the agent make the right calls?",
     "A fake model plays a script: invented quotes, short answers, API errors,\nbudget overruns, a contradictory brief. Check every route and every label.")
step(0)
cmd("python -m evals.trajectory")
for r in traj["results"]:
    ok = r["passed"]
    emit([("PASS  " if ok else "FAIL  ", mono_b, TEAL if ok else RED), (SCENARIO_EN.get(r["scenario"], r["scenario"]), mono, TXT)], 0.42)
blank()
emit([(f"{traj['passed']}/{traj['scenarios']} scenarios · routes {traj['path_correct']}/{traj['scenarios']} · "
       f"field labels {traj['status_correct']} · trust-rule violations {traj['invariant_violations']}", mono_b, LIME)], 1.0)
note("Trust rules checked in every run: 'pass' needs verified evidence; calls and spend are capped;", 0.6)
note("nothing is delivered without a person confirming it.", 2.6)

# ---- 2 mutation ----
term.clear()
card("STEP 2", "Break it on purpose", "An eval that always passes tests nothing.\nSwitch off one guardrail at a time and rerun the scenarios.")
step(1)
cmd("python -m evals.mutation")
for r in mut["results"]:
    emit([("CAUGHT  " if r["caught"] else "MISSED  ", mono_b, TEAL if r["caught"] else RED), (f"{r['mutant']:<36}", mono, TXT),
          (f"{len(r['failed_scenarios'])} scenario(s) fail", mono, DIM)], 0.8)
blank()
emit([(f"{mut['caught']}/{mut['mutants']} broken guardrails caught", mono_b, LIME)], 2.6)

# ---- 3 parity ----
term.clear()
card("STEP 3 · 23 REAL CLIENT BRIEFS", "Did shipping it change quality?",
     "Same briefs, same model, same day: the experiment's code path vs the shipped agent.\nScored with the experiment's own scorers. Allowed drop: 0.036.")
step(2)
cmd("python -m evals.parity")
head(f"{par['briefs']} briefs · experiment path vs shipped agent · 95% CI resampled by brief")
o = par["overall_sentence"]
emit([(f"{'field':<15}{'experiment':>11}{'shipped':>10}{'diff':>9}   {'95% CI':<18}", mono_b, DIM)], 0.5, 18)
for t, r in par["per_task"].items():
    v, c = verdict(*r["ci95"])
    emit([(f"{TASK_EN[t]:<15}{r['experiment_arm']:>11.3f}{r['production_arm']:>10.3f}{r['diff']:>+9.3f}   "
           f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]  ", mono, TXT), (v, mono_b, c)], 0.4, 18)
blank()
v, c = verdict(*o["ci95"])
emit([("8 sentence fields: ", mono_b, TXT), (f"{o['diff']:+.3f}  [{o['ci95'][0]:+.3f}, {o['ci95'][1]:+.3f}]  ", mono, TXT), (v, mono_b, c)], 1.2)
note("8 sentence fields: no quality loss after shipping. Keywords: not yet proven, still being fixed.", 1.4)
note("The experiment's own scores reproduce today: the model has not drifted.", 2.4)

term.clear()
step(2)
head("What the first run of this eval found")
kb, ka = par_before["per_task"]["keywords"], par["per_task"]["keywords"]
emit([("keywords before fix:  ", mono_b, TXT), (f"{pct(kb['production_coverage'])} of briefs got keywords, F1 {kb['production_arm']:.3f}  ", mono, TXT),
      ("worse", mono_b, RED)], 1.2, 18)
note("Cause: the output schema said 'comma-separated', the prompt said 'numbered list'.", 1.0)
note("The model followed either one; the parser only read numbered lists.", 1.6)
emit([("keywords after fix:   ", mono_b, TXT), (f"{pct(ka['production_coverage'])} of briefs got keywords, F1 {ka['production_arm']:.3f}", mono, LIME)], 1.4, 18)
note("Also fixed: list fields reached the prompt as Python text ['A', 'B'].", 1.4)
note("Neither bug was caught by unit tests.", 2.6)

# ---- 4 checkers ----
term.clear()
card("STEP 4", "Can the agent's 'pass' label be trusted?",
     f"{chk['n_fields']} fact fields from the real briefs; {chk['n_bad']} are weak (similarity to the reference < 0.60).\nFind a signal that separates good drafts from weak ones.")
step(3)
cmd("python -m evals.checkers")
cr = chk["current_router"]
emit([("current 'pass' label:   ", mono_b, TXT), (f"AUC {cr['auc']:.2f}  ", mono_b, RED), ("(0.50 = coin toss)", mono, DIM)], 1.6, 18)
blank()
emit([(f"{'candidate checker':<36}{'AUC':>6}   {'weak drafts caught when people review 30%':<44}", mono_b, DIM)], 0.6, 18)
for name in ["current_overlap", "A_self_consistency", "C_source_similarity", "B_ai_judge"]:
    m = chk["checkers"][name]
    emit([(f"{CHECKER_EN[name]:<36}", mono, TXT), (f"{m['auc']:>6.2f}", mono_b, TEAL if m["auc"] >= 0.7 else AMBER),
          (f"   {pct(m['catch_recall_at_budget'])}", mono, TXT)], 0.7, 18)
blank()
dist = chk["judge_score_distribution"]
note(f"The AI judge gave every field a {' or '.join(str(k) for k in dist)}: too lenient to use.", 1.3)
note("Self-consistency is not significantly better and costs 2 extra calls per field.", 1.3)
note("Pick word overlap: free, already computed. Only its threshold was wrong.", 2.6)

# ---- 5 fix ----
term.clear()
step(4)
rb = chk["router_before_after"]
head(f"Threshold {rb['before_threshold']} → {rb['after_threshold']}   (validated leave-one-brief-out)")
b, a = rb["before"], rb["after_leave_one_out"]
emit([(f"{'':<34}{'before':>8}{'after':>8}", mono_b, DIM)], 0.5, 18)
for label, key, good_up in [("fields sent to a person", "flag_rate", None), ("weak drafts caught", "catch_recall", True),
                            ("weak drafts missed", "miss_rate", False), ("auto-passed drafts that are good", "auto_pass_good_rate", True)]:
    emit([(f"{label:<34}", mono, TXT), (f"{pct(b[key]):>8}", mono, DIM), (f"{pct(a[key]):>8}", mono_b, LIME if good_up is not None else TXT)], 0.9, 18)
blank()
note(f"Still misses {pct(a['miss_rate'])} of weak drafts, so a person still reviews every field.", 1.6)
note("Next: human labels on 200+ fields to test the release bar (miss ≤ 5%, false block ≤ 20%).", 2.8)

state["done"] |= {4}
cost = par["ops"]["production_cost_usd"] + par["ops"]["experiment_cost_usd"] + chk["spent_usd"]
card("THE TAKEAWAY", "Measured, then improved.",
     f"No quality loss on the 8 sentence fields · 2 production bugs found and fixed\n'Pass' label: coin toss → catches {pct(a['catch_recall'])} of weak drafts\n"
     f"All three evals together cost about ${cost:.2f}", 5.0)

w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
for fr in frames: w.append_data(fr)
w.close()
print("frames", len(frames), "seconds", round(len(frames) / FPS, 1))
