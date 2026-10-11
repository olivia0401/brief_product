"""Render a terminal-style replay video of the real demo run (logs in brief_product/demo/*.log).

Console labels are shown in English; model output, evidence, costs and file names are copied from the logs.
Grey italic lines marked '·' are narration, not program output.
"""
import sys
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 15
OUT = sys.argv[1]
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
TEAL, LIME, RED, SLATE, WHITE = "#4ECDC4", "#C7F464", "#FF7B72", "#454F5B", "#FFFFFF"
STEPS = ["Read brief", "AI drafts x9", "Ask client  (pause)", "Human review  (pause)", "Deliver"]

frames = []
def hold(img, secs):
    a = np.asarray(img.convert("RGB"))
    frames.extend([a] * max(1, int(secs * FPS)))

# ---------- title cards ----------
def card(kicker, title, sub, secs=3.2):
    im = Image.new("RGB", (W, H), SLATE); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=LIME)
    d.text((110, 220), kicker, font=ui_b, fill=LIME)
    d.text((106, 250), title, font=big, fill=WHITE)
    y = 340
    for line in sub.split("\n"):
        d.text((110, y), line, font=mid, fill="#DDE3E8"); y += 38
    d.rectangle([110, 200, 200, 205], fill=TEAL)
    d.text((110, H - 60), "Replay of a real run · fictional sample brief · real GPT-5-mini output", font=small, fill="#B9C2CB")
    hold(im, secs)

# ---------- terminal ----------
TOP, LH, LEFT, MAXW = 132, 27, 40, W - 80
term = []                      # list of rendered lines: list of (text, font, colour)

def wrap(segs, indent=0):
    """segs: [(text, font, colour)] → list of lines, wrapped to the terminal width."""
    lines, cur, x = [], [], indent
    for text, font, colour in segs:
        for word in text.split(" "):
            token = (word + " ")
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
    d.text((100, 12), "brief_product — demo_run.py   (replay of a real run)", font=ui, fill="#C9D1D9")
    # step chips: where we are in the LangGraph flow
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
    visible = term[-int((H - TOP - 40) / LH):]
    y = TOP
    for indent, segs in visible:
        xx = LEFT + indent
        for text, font, colour in segs:
            d.text((xx, y), text, font=font, fill=colour); xx += font.getlength(text)
        y += LH
    d.text((40, H - 30), "Console labels translated to English. Grey '·' lines are narration, not program output.", font=ui, fill="#6E7A86")
    return im

state = {"step": 0, "done": set()}
def emit(segs, secs=0.45, indent=0):
    for line in wrap(segs, indent):
        term.append(line)
        hold(render(state["step"], state["done"]), secs)
def blank(): term.append((0, [("", mono, TXT)]))
def step(i):
    state["done"] |= set(range(i)); state["step"] = i

def cmd(c): emit([("$ ", mono_b, LIME), (c, mono, TXT)], 0.9)
def head(t): emit([("» ", mono_b, TEAL), (t, mono_b, TXT)], 0.8)
def kv(k, v, hl=False, secs=0.45): emit([(k + ": ", mono_b, LIME if hl else DIM), (v, mono, LIME if hl else TXT)], secs, 18)
def note(t, secs=1.2): emit([("· " + t, mono_i, DIM)], secs, 18)
def pause(t): blank(); emit([(" PAUSED ", mono_b, "#1B2128"), ("  " + t, mono_b, LIME)], 1.4)

# fix the PAUSED tag background: draw it as lime text on dark with brackets instead
def pause(t):
    blank(); emit([("[PAUSED] ", mono_b, LIME), (t, mono_b, LIME)], 1.4)

# ================== the story ==================
card("LANGGRAPH IN ACTION", "My Monika brief workflow", "A real run, replayed: AI drafts, the flow pauses for people,\nand picks up where it stopped days later.", 4.0)

card("DAY 1", "The client sends a brief", "One key field is missing. Watch what the workflow does.")
step(0)
cmd("python demo_run.py start")
head("Day 1: client submits the brief")
kv("Business category", "Consumer electronics, smart home devices")
kv("Product", "A line of voice-controlled smart home hubs that connect lights, locks and thermostats. Sets up in under five minutes.")
kv("Differentiators", "Faster setup, lower price, no subscription required")
kv("Brand strategy", "Friendly tech that respects your time. Approachable, low-fuss, with a sense of humour.")
kv("Personality", "Approachable, witty, reliable", secs=0.8)
step(1)
note("9 sections drafted in parallel, each checked by rules", 1.6)
step(2)
pause("Waiting for the client to reply")
kv("Missing key field", "['audience']", secs=0.8)
kv("AI-drafted clarification email", "", secs=0.4)
emit([("\"Please complete and return your brand brief at your earliest convenience—we're missing the audience section and need it to proceed; if you have questions or prefer a quick call, I'm happy to help.\"", mono_i, TXT)], 0.5, 36)
hold(render(state["step"], state["done"]), 2.5)
note("The program stops here. Progress is saved to the SQLite checkpoint.", 2.5)

term.clear()
card("DAY 3", "A brand-new process starts", "It finds the saved checkpoint and continues from exactly where it stopped.")
step(2)
cmd("python demo_run.py client_replied")
head("Day 3: new process, find where we stopped in the checkpoint")
kv("Checkpoint stopped at", "['ask_client']", hl=True, secs=1.4)
kv("Manager sent the email; client added", "", secs=0.6)
kv("+ Audience", "Mid-30s renters in urban apartments who want connected-home convenience without a subscription bill.", hl=True, secs=1.6)
step(1)
note("Brief read again; 9 sections re-drafted in parallel", 1.4)
step(3)
pause("Waiting for the manager to check each section")
rows = [("concept", "suggestion", "$0.00043"), ("position", "suggestion", "$0.00023"), ("emotion", "suggestion", "$0.00026"),
        ("function", "passed", "$0.00020"), ("benefit", "passed", "$0.00022"),
        ("category", "format failed (14 words, needs 15-20)", "$0.00039"),
        ("feature", "passed", "$0.00022"), ("context", "passed", "$0.00019"), ("keywords", "passed", "$0.00067")]
for name, st, cost in rows:
    c = TEAL if st == "passed" else (RED if "failed" in st else "#E3B341")
    emit([(f"{name:<10}", mono_b, TXT), (f"{st:<40}", mono, c), (cost, mono, DIM)], 0.35, 18)
blank()
emit([("feature draft: ", mono_b, DIM), ("\"…centralizes control, enhances automation and ", mono, TXT), ("security", mono_b, RED), (".\"", mono, TXT)], 1.6, 18)
note("Passed the automatic check, but the brief never mentions security.", 3.0)

term.clear()
card("REVIEW", "A person makes the final call", "Another new process resumes the same task and delivers it.")
step(3)
cmd("python demo_run.py review")
head("Manager review: new process, resume from checkpoint")
kv("Decisions", "accept x6, edit x3 (category, feature, context)", secs=1.4)
kv("feature, after edit", "\"Voice-controlled smart home hubs that connect lights, locks and thermostats, with setup that takes under five minutes.\"", hl=True, secs=2.0)
step(4)
blank()
emit([("[DONE] Delivered: ", mono_b, TEAL), ("data/demo-acme/confirmed/acme_v1-1dc615c9.json", mono, TXT)], 1.6)
kv("Every section", "status \"confirmed\", with quotes from the brief as evidence", secs=1.2)
kv("Total cost (two drafting rounds + email)", "$0.005375", hl=True, secs=3.0)
state["done"] |= {4}

card("THE TAKEAWAY", "Three runs. One task.", "AI drafted, rules checked, people decided.\nLangGraph remembered where it stopped.", 4.5)

w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
for fr in frames: w.append_data(fr)
w.close()
print("frames", len(frames), "seconds", round(len(frames) / FPS, 1))
