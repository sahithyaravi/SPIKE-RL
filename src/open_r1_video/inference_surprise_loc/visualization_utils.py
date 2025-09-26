
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


import os

from IPython.display import HTML
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from typing import List

import seaborn as sns
from typing import List
import re

import math, textwrap, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import numpy as np, textwrap
from matplotlib import cm
from moviepy import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

from pathlib import Path
from typing import List, Sequence, Tuple

def wrap_text(text, font, max_width):
    """Word-wrap a single paragraph to fit pixel width."""
    words, lines, line = text.split(), [], ""
    for w in words:
        test = f"{line} {w}".strip()
        if font.getbbox(test)[2] <= max_width:
            line = test
        else:
            lines.append(line)
            line = w
    lines.append(line)
    return lines


def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def visualize_surprise_heatmap(scores: dict, top_k: int = 3):
    data = np.array([scores["word_shift"], scores["semantic_shift"]])
    fig, ax = plt.subplots(figsize=(12, 2))
    sns.heatmap(data, cmap="viridis", cbar=True, xticklabels=False, 
                yticklabels=["Word Shift", "Semantic Shift"], ax=ax)
    plt.title("Surprise Metrics Heatmap Across Frames")
    plt.xlabel("Frame Index")
    plt.show()

    top_indices = np.argsort(scores["semantic_shift"])[-top_k:]
    print(f"Top {top_k} surprising frames (semantic): {top_indices.tolist()}")
    return top_indices.tolist()

def show_surprising_frames(frames: List[np.ndarray], scores: List[float], top_k: int = 10):
    print(len(frames), len(scores))
    top_indices = np.argsort(scores)[-top_k:]
    print(f"Top {top_k} surprising frames (semantic): {top_indices.tolist()}")
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(top_indices):
        plt.subplot(1, top_k, i + 1)
        plt.imshow(frames[idx])
        plt.title(f"{scores[idx]:.2f}")
        plt.axis("off")
    plt.suptitle("Most Surprising Frames")
    plt.show()


def display_video_from_url(url: str):
    return HTML(f"""
    <video width="640" height="360" controls>
      <source src="{url}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """)


def display_local_video(video_path: str, width: int = 640, height: int = 360):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found at: {video_path}")
    
    return HTML(f"""
    <video width="{width}" height="{height}" controls>
      <source src="{video_path}" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    """)




# --- up-scaling -----------------------------------------------------------


def upscale_to_1080p(image, target_height=1080):
    H, W = image.shape[:2]
    scale = target_height / H
    new_size = (int(W * scale), target_height)
    img_resized = Image.fromarray(image).resize(new_size, Image.BICUBIC)
    
    # Pad to exactly 1920x1080 if needed
    pad_w = 1920 - new_size[0]
    if pad_w > 0:
        padded = Image.new("RGB", (1920, 1080), (255, 255, 255))
        padded.paste(img_resized, (pad_w // 2, 0))
        return np.array(padded)
    else:
        return np.array(img_resized)


# --------------------------------------------------------------------------------------
#  CONFIG
# --------------------------------------------------------------------------------------
FONT_PATH = Path("/Users/sahithyaravi/Documents/BlackSwanAdapt/DejaVuSans.ttf")
TARGET_W  = 1920       # final video width
BAR_H     = 60         # spark‑line bar height (heat‑bar currently disabled)
CURSOR_W  = 3          # moving cursor width (px)
TOP_K     = 8          # max hypotheses shown per panel
FPS       = 3

JS_RE = re.compile(r"JS:\s*([+-]?[0-9.]+)")

# --------------------------------------------------------------------------------------
#  UTIL: simple greedy word‑wrap (unchanged)
# --------------------------------------------------------------------------------------

def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> List[str]:
    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    words = text.split()
    lines: List[str] = []
    cur = ""
    for w in words:
        trial = (cur + " " + w).strip()
        if draw.textbbox((0, 0), trial, font=font)[2] <= max_w:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

# --------------------------------------------------------------------------------------
#  BUILD‑EXPLANATION PANEL  (black text, coloured background)
# --------------------------------------------------------------------------------------

def build_expl_panel(
    expl: Sequence[str],
    W: int,
    font: ImageFont.FreeTypeFont,
    *,
    bullet: str = "•",
    pad: int = 10,
    gap: int = 4,
    bgcolor: Tuple[int, int, int] = (255, 255, 255),
    memory: str = "",
):
    """Render hypotheses list with ΔJS‑based *background* colour.

    Each entry in *expl* must include "JS: ±0.08".  We sort by |ΔJS| (top‑k), draw
    a full‑width coloured rectangle per line, then overlay bullet + text in
    **black** for readability.
    """
    # parse & sort by |ΔJS|
    parsed = []
    for item in expl:
        m = JS_RE.search(item)
        d_js = float(m.group(1)) if m else 0.0
        parsed.append((item, d_js))
    if parsed:
        max_abs = max(abs(d) for _, d in parsed) or 1e-6
        parsed.sort(key=lambda x: -abs(x[1]))
        parsed = parsed[:TOP_K]
    else:
        parsed, max_abs = [("<no hypotheses>", 0.0)], 1.0

    cmap = cm.get_cmap("coolwarm")

    dummy = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    bullet_w = dummy.textbbox((0, 0), f"{bullet} ", font=font)[2]
    max_text_w = W - 2 * pad - bullet_w

    # flatten wrapped lines
    lines, first_flags, bg_colours = [], [], []
    for text, d_js in parsed:
        frac = 0.5 + 0.5 * (d_js / max_abs)
        colour = tuple(int(c * 255) for c in cmap(frac)[:3])
        wrapped = wrap_text(text, font, max_text_w)
        lines.extend(wrapped)
        first_flags.extend([True] + [False] * (len(wrapped) - 1))
        bg_colours.extend([colour] * len(wrapped))

    if memory:
        mem_lines = wrap_text("History: " + memory, font, W - 2 * pad)
        lines.extend(mem_lines)
        first_flags.extend([False] * len(mem_lines))
        bg_colours.extend([(240, 240, 240)] * len(mem_lines))

    line_h = font.getbbox("Hg")[3]
    panel_h = pad + len(lines) * (line_h + gap)
    panel   = Image.new("RGB", (W, panel_h), bgcolor)
    draw    = ImageDraw.Draw(panel)

    y = pad // 2
    for txt, first, colour in zip(lines, first_flags, bg_colours):
        # background strip
        draw.rectangle([(0, y - 2), (W, y + line_h + 2)], fill=colour)
        # bullet + black text
        if first:
            draw.text((pad, y), f"{bullet} {txt}", font=font, fill=(0, 0, 0))
        else:
            draw.text((pad + bullet_w, y), txt, font=font, fill=(0, 0, 0))
        y += line_h + gap

    return panel, panel_h

# --------------------------------------------------------------------------------------
#  MAIN ENTRY – make_scrollbar_video
# --------------------------------------------------------------------------------------

def make_scrollbar_video(
    frames: Sequence[np.ndarray | Image.Image],
    scores: Sequence[float],
    explanations: Sequence[Sequence[str]],
    *,
    out: str = "surprise_timeline.mp4",
    amusing_indices: Sequence[int] | None = None,
    memory: str | None = None,
):
    """Compose MP4 with raw frame, *spark‑line only*, and coloured‑background panel."""

    assert len(frames) == len(scores) == len(explanations), "length mismatch"
    N = len(frames)
    amusing_indices = set(amusing_indices or [])

    font_big   = ImageFont.truetype(str(FONT_PATH), 28)
    font_small = ImageFont.truetype(str(FONT_PATH), 18)

    # normalise scores
    scores_arr = np.asarray(scores, dtype=float)
    s_min, s_max = float(scores_arr.min()), float(scores_arr.max())
    scores_norm = (scores_arr - s_min) / (s_max - s_min + 1e-9)

    # scale first frame to get target_h
    first = frames[0] if isinstance(frames[0], Image.Image) else Image.fromarray(frames[0])
    raw_w, raw_h = first.size
    target_h = int(raw_h * TARGET_W / raw_w)

    step_w = TARGET_W / N

    # ――― spark‑line points
    spark_pts = [(
        int(i * step_w + step_w / 2),
        BAR_H - 4 - int(scores_norm[i] * (BAR_H - 8))
    ) for i in range(N)]

    # resize helper
    def _resize(img: Image.Image) -> Image.Image:
        return img.resize((TARGET_W, target_h), Image.Resampling.LANCZOS)

    # PASS 1: build & pad all panels
    panels, heights = [], []
    for expl in explanations:
        p, h = build_expl_panel(expl, TARGET_W, font_small, memory=memory or "")
        panels.append(p); heights.append(h)
    max_panel_h = max(heights)
    padded_panels = [(
        p if h == max_panel_h else Image.new("RGB", (TARGET_W, max_panel_h), "white").paste(p, (0, 0)) or p
    ) for p, h in zip(panels, heights)]

    FIXED_H = target_h + BAR_H + max_panel_h

    # PASS 2: compose frames
    video_frames: List[np.ndarray] = []
    for idx, (raw, s, panel) in enumerate(zip(frames, scores, padded_panels)):
        img = raw if isinstance(raw, Image.Image) else Image.fromarray(raw)
        img = _resize(img)

        if idx in amusing_indices:
            d = ImageDraw.Draw(img)
            d.rectangle([(6, 6), (TARGET_W - 6, target_h - 6)], outline="red", width=8)
            d.text((14, 14), "🤣", font=font_big, fill="red")

        # blank (white) bar
        bar = Image.new("RGB", (TARGET_W, BAR_H), "white")
        db  = ImageDraw.Draw(bar)
        # spark‑line
        db.line(spark_pts[: idx + 1], fill=(0, 0, 0), width=3)
        x_cur = spark_pts[idx][0]
        db.rectangle([(x_cur - CURSOR_W // 2, 0), (x_cur + CURSOR_W // 2, BAR_H)], fill="black")
        # numeric score
        txt = f"{s:.2f}"
        tw, th = db.textbbox((0, 0), txt, font=font_big)[2:4]
        db.text((x_cur - tw // 2, (BAR_H - th) // 2), txt, font=font_big, fill="black")

        # stack
        canvas = Image.new("RGB", (TARGET_W, FIXED_H), "white")
        canvas.paste(img,   (0, 0))
        canvas.paste(bar,   (0, target_h))
        canvas.paste(panel, (0, target_h + BAR_H))
        video_frames.append(np.asarray(canvas))

    # write video
    # try:
    ImageSequenceClip(video_frames, fps=FPS).write_videofile(
        out,
        codec="libx264",
        audio=False,
        bitrate="8M",
        preset="slow",
        ffmpeg_params=[
            "-pix_fmt", "yuv420p",      # essential for player compatibility
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"
        ]
        
    )
    print("Saved →", out)
    # except:
    #     # just write no ffmpeg video
    #     print("FFmpeg failed, writing no ffmpeg video")
    #     ImageSequenceClip(video_frames, fps=FPS).write_videofile(
    #         out,
    #         codec="libx264",
    #         audio=False,
    #         bitrate="8M",
    #         preset="slow"
    #     )
    #     print("Saved →", out)