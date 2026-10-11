"""One video for the whole case: the agent at work (replay of a real run), then how it was evaluated.

    python demo/make_full_demo.py demo/full_demo.mp4

Renders part 1 with make_replay.py --part1 and part 2 with make_eval_replay.py --part2, then joins them with an
opening card and a bridge card. The bridge picks up a moment from part 1: a field that passed the automatic check
but added a claim the brief never made, which is the question part 2 answers.
"""
import subprocess, sys, tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = sys.argv[1]
HERE = Path(__file__).resolve().parent
W, H, FPS = 1280, 720, 15
F = "C:/Windows/Fonts/"
ui_b = ImageFont.truetype(F + "seguisb.ttf", 15)
big = ImageFont.truetype(F + "seguisb.ttf", 54)
mid = ImageFont.truetype(F + "segoeui.ttf", 26)
small = ImageFont.truetype(F + "segoeui.ttf", 18)
LIME, TEAL, SLATE, WHITE = "#C7F464", "#4ECDC4", "#454F5B", "#FFFFFF"


def card(kicker, title, sub, footer):
    im = Image.new("RGB", (W, H), SLATE); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, 14, H], fill=LIME)
    d.rectangle([110, 200, 200, 205], fill=TEAL)
    d.text((110, 220), kicker, font=ui_b, fill=LIME)
    d.text((106, 250), title, font=big, fill=WHITE)
    y = 340
    for line in sub.split("\n"):
        d.text((110, y), line, font=mid, fill="#DDE3E8"); y += 38
    d.text((110, H - 60), footer, font=small, fill="#B9C2CB")
    return np.asarray(im)


with tempfile.TemporaryDirectory() as tmp:
    p1, p2 = Path(tmp) / "part1.mp4", Path(tmp) / "part2.mp4"
    subprocess.run([sys.executable, str(HERE / "make_replay.py"), str(p1), "--part1"], check=True)
    subprocess.run([sys.executable, str(HERE / "make_eval_replay.py"), str(p2), "--part2"], check=True)

    w = imageio.get_writer(OUT, fps=FPS, codec="libx264", quality=8, macro_block_size=8, pixelformat="yuv420p")
    def hold(frame, secs):
        for _ in range(int(secs * FPS)):
            w.append_data(frame)
    def stream(path):
        n = 0
        for frame in imageio.get_reader(str(path)):
            w.append_data(frame); n += 1
        return n

    hold(card("CASE STUDY · LANGGRAPH AGENT", "Brand-brief extraction agent",
              "Turns long client brand briefs into checked inputs for a naming team.\n"
              "Part 1 · watch it run, start to finish.\nPart 2 · see how it was evaluated, and what that changed.",
              "Real GPT-5-mini runs · the demo brief is fictional · real client briefs stay private"), 5.0)
    n1 = stream(p1)
    hold(card("PART 2 · EVALUATING THE AGENT", "But was it right?",
              "In that run, 'feature' passed every automatic check, yet it said 'security',\n"
              "which the brief never mentions. A person caught it. So: can the 'pass' label\n"
              "be trusted, and did shipping the agent change its quality? Measure it.",
              "Replay of real eval runs · only aggregate numbers are shown"), 6.0)
    n2 = stream(p2)
    w.close()

total = 5.0 * FPS + n1 + 6.0 * FPS + n2
print("seconds", round(total / FPS, 1))
