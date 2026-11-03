"""Three 1280x720 YouTube thumbnails from the best frames, short punchy text and an AI background.

make_thumbnails(shotlist, titles, ctx) -> [thumb_1.png, thumb_2.png, thumb_3.png] in ctx.root/thumbnails.
1: best frame, boosted contrast and saturation, dark gradient on the side away
   from the subject, huge stroked text with a drop shadow.
2: another frame zoomed to its most detailed region, bold colour block with
   text and an arrow toward the subject.
3: AI image of the title concept (image seat) with centred text and an accent bar.
Text is Inter Black, uppercase, at most 2 lines, sized as large as fits.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps

from ..models import ShotList, SourceKind
from ..util import no_em_dash, run, slug
from .cache import resolve_media

SIZE = (1280, 720)
ACCENT = (255, 216, 74)
ARROW = (235, 40, 40)
MAX_WORDS = 4
MAX_CHARS = 24  # longer texts get too small to read on a phone
STOP = {"the", "a", "an", "of", "to", "and", "in", "on", "at", "for", "with", "by", "from", "that", "this",
        "how", "why", "what", "is", "was", "it", "its", "into", "they", "who", "really", "still"}
SYSTEM = ("You write YouTube thumbnail text. Two to four punchy words each, uppercase, accurate, no clickbait, "
          "no emojis, no em dashes. Reply with strict JSON only.")


def make_thumbnails(shotlist: ShotList, titles: list[str], ctx: Any) -> list[Path]:
    """Write thumb_1..3.png (1280x720) and return their paths."""
    out_dir = ctx.root / "thumbnails"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = best_frames(shotlist, ctx, 5)
    texts = thumbnail_texts(titles, ctx)
    font = font_path(ctx.settings)
    concept = titles[0] if titles else ctx.project.title
    ai_bg = _ai_background(concept, ctx, out_dir)
    first = frames[0] if frames else ai_bg
    second = frames[1] if len(frames) > 1 else first
    images = [comp_gradient(_load(first), texts[0], font),
              comp_subject(_load(second), texts[1], font),
              comp_centered(_load(ai_bg), texts[2], font)]
    paths = []
    for i, im in enumerate(images, 1):
        p = out_dir / f"thumb_{i}.png"
        im.convert("RGB").save(p, "PNG", optimize=True)
        paths.append(p)
    ctx.log.info("thumbnails written", count=len(paths), frames=len(frames), texts=texts)
    return paths


def font_path(settings: Any) -> Path:
    return settings.path("assets/fonts/Inter-Black.otf")


# ---------------------------------------------------------------- frames

def best_frames(shotlist: ShotList, ctx: Any, k: int) -> list[Path]:
    """Up to k distinct frame images from selected candidates, best judge score first (motion graphics last)."""
    scored: list[tuple[int, float, Path]] = []
    seen: set[str] = set()
    grab_dir = ctx.work / "thumbs"
    for shot in shotlist.shots:
        c = shot.selected_candidate()
        if c is None or c.id in seen:
            continue
        seen.add(c.id)
        img = _frame_for(shot, c, ctx, grab_dir)
        if img is None:
            continue
        score = c.score if c.score is not None else shot.judge_score
        if score is None:
            res = ctx.seats.vision_json("judge", "judge", "Score this frame as a YouTube thumbnail, 0 to 100.",
                                        shot.intent, [str(img)], stage="packaging", intent=shot.intent)
            score = float(res.get("score", 0) or 0)
        scored.append((1 if c.source == SourceKind.motion else 0, -float(score), img))
    scored.sort(key=lambda t: (t[0], t[1]))
    return [p for _, _, p in scored[:k]]


def _frame_for(shot: Any, c: Any, ctx: Any, grab_dir: Path) -> Optional[Path]:
    """Keyframe (middle one for video), the still itself, or a frame grabbed from the clip or segment."""
    keys = [resolve_media(p, ctx.settings) for p in c.keyframes if p]
    keys = [p for p in keys if p.is_file()]
    if keys:
        return keys[len(keys) // 2]
    src = resolve_media(c.local_path, ctx.settings) if c.local_path else None
    if src is not None and src.is_file() and c.media == "image":
        return src
    t = (c.in_point if src is not None and src.is_file() else 0.0) + shot.duration / 2
    for video in (src, Path(shot.render_segment_path) if shot.render_segment_path else None):
        if video is not None and video.is_file():
            out = grab_dir / f"{slug(shot.id)}.png"
            grab_dir.mkdir(parents=True, exist_ok=True)
            try:
                run(["ffmpeg", "-y", "-v", "error", "-nostdin", "-ss", f"{t:.3f}", "-i", str(video),
                     "-frames:v", "1", str(out)], timeout=60)
            except Exception as e:  # noqa: BLE001 - an unreadable clip just gives no candidate frame
                ctx.log.warn("thumbnail frame grab failed", shot=shot.id, error=str(e)[:200])
                continue
            if out.exists():
                return out
            t = shot.duration / 2
    return None


def _ai_background(concept: str, ctx: Any, out_dir: Path) -> Path:
    prompt = (f"Cinematic, dramatic documentary still illustrating: {concept}. Strong subject, bold lighting, "
              f"rich colour, high contrast, no text, no logos.")
    path, _mock = ctx.seats.image(prompt, ctx.work / "thumbs" / "ai_background.png", stage="packaging")
    return Path(path)


def _load(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


# ---------------------------------------------------------------- text

def thumbnail_texts(titles: list[str], ctx: Any) -> list[str]:
    """Three short uppercase texts (writer seat task thumbnail_text, heuristic fallback)."""
    user = ("Titles:\n" + "\n".join(f"- {t}" for t in titles) +
            '\nReturn {"texts": [3 different thumbnail texts, 2 to 4 words each]}.')
    data = ctx.seats.llm_json("writer", "thumbnail_text", SYSTEM, user, stage="packaging")
    texts: list[str] = []
    if isinstance(data, dict):
        for t in data.get("texts") or []:
            t = _short(t) if isinstance(t, str) else ""
            if t and t not in texts:
                texts.append(t)
    for t in (heuristic_text(x) for x in titles or [ctx.project.title]):
        if t and t not in texts:
            texts.append(t)
    while len(texts) < 3:
        texts.append(texts[-1] if texts else "WATCH THIS")
    return texts[:3]


def _short(text: str) -> str:
    """Uppercase text of at most MAX_WORDS words and MAX_CHARS characters, else ''."""
    words = re.sub(r"\s+", " ", no_em_dash(text)).strip().upper().split()
    out = " ".join(words)
    return out if 1 <= len(words) <= MAX_WORDS and len(out) <= MAX_CHARS else ""


def heuristic_text(title: str) -> str:
    """2 to 4 strong words from a title: a number phrase, else the subtitle, else the topic."""
    title = no_em_dash(title)
    m = re.search(r"\b\d[\d,.]*\b(?:\s+[A-Za-z]+){0,2}", title)
    if m and len(m.group(0).split()) >= 2:
        return _short(m.group(0))
    parts = [p.strip() for p in title.split(":") if p.strip()]
    for part in reversed(parts):
        words = [w for w in re.findall(r"[A-Za-z0-9']+", part)]
        while words and words[0].lower() in STOP:
            words.pop(0)
        if 2 <= len(words) <= MAX_WORDS and _short(" ".join(words)):
            return _short(" ".join(words))
    strong = [w for w in re.findall(r"[A-Za-z0-9']+", title) if w.lower() not in STOP]
    while len(strong) > 1 and not _short(" ".join(strong[:3])):
        strong.pop()
    return _short(" ".join(strong[:3]))


def split_lines(text: str) -> list[list[str]]:
    """Candidate layouts: one line, then the most balanced two-line split."""
    words = text.split()
    options = [[text]]
    if len(words) > 1:
        best = min(range(1, len(words)), key=lambda i: abs(len(" ".join(words[:i])) - len(" ".join(words[i:]))))
        options.append([" ".join(words[:best]), " ".join(words[best:])])
    return options


def fit_text(text: str, font: Path, max_w: int, max_h: int, top: int = 190) -> tuple[list[str], ImageFont.FreeTypeFont]:
    """Largest Inter Black size and 1 or 2 line layout whose block fits max_w x max_h (one line preferred)."""
    best: tuple[float, int, list[str]] = (0.0, 30, [text])
    for lines in split_lines(text):
        for size in range(top, 30, -4):
            f = ImageFont.truetype(str(font), size)
            w = max(f.getlength(ln) for ln in lines) + size * 0.16
            if w <= max_w and len(lines) * size * 1.02 <= max_h:
                rank = size * (1.1 if len(lines) == 1 else 1.0)
                if rank > best[0]:
                    best = (rank, size, lines)
                break
    return best[2], ImageFont.truetype(str(font), best[1])


def draw_text(im: Image.Image, lines: list[str], font: ImageFont.FreeTypeFont, x: int, y: int,
              fills: list[tuple[int, int, int]], align: str = "left") -> tuple[int, int, int, int]:
    """Stroked text with a soft drop shadow; returns the text block box."""
    size = font.size
    stroke = max(4, int(size * 0.075))
    line_h = int(size * 1.02)
    widths = [font.getlength(ln) for ln in lines]
    block_w = int(max(widths))
    shadow = Image.new("L", im.size, 0)
    sd = ImageDraw.Draw(shadow)
    pos = []
    for i, (ln, w) in enumerate(zip(lines, widths)):
        lx = x + (block_w - w) / 2 if align == "center" else x
        pos.append((lx, y + i * line_h))
        sd.text((lx + size * 0.05, y + i * line_h + size * 0.07), ln, font=font, fill=200,
                stroke_width=stroke)
    shadow = shadow.filter(ImageFilter.GaussianBlur(max(3, size // 18)))
    im.paste(Image.new("RGB", im.size, (0, 0, 0)), (0, 0), shadow)
    d = ImageDraw.Draw(im)
    for i, (ln, p) in enumerate(zip(lines, pos)):
        d.text(p, ln, font=font, fill=fills[min(i, len(fills) - 1)], stroke_width=stroke, stroke_fill=(0, 0, 0))
    return x, y, x + block_w, y + line_h * len(lines)


# ---------------------------------------------------------------- image helpers

def cover(im: Image.Image, size: tuple[int, int] = SIZE, centering: tuple[float, float] = (0.5, 0.5)) -> Image.Image:
    return ImageOps.fit(im, size, method=Image.LANCZOS, centering=centering)


def punch(im: Image.Image, contrast: float = 1.22, colour: float = 1.38) -> Image.Image:
    """Thumbnail grade: contrast, saturation and a little sharpening."""
    im = ImageEnhance.Contrast(im).enhance(contrast)
    im = ImageEnhance.Color(im).enhance(colour)
    return ImageEnhance.Sharpness(im).enhance(1.25)


def energy(im: Image.Image) -> np.ndarray:
    """Detail and saturation map used to locate the subject."""
    a = np.asarray(im.convert("RGB").resize((160, 90)), dtype=np.float32) / 255.0
    lum = a.mean(axis=2)
    gx = np.abs(np.diff(lum, axis=1, prepend=lum[:, :1]))
    gy = np.abs(np.diff(lum, axis=0, prepend=lum[:1, :]))
    sat = a.max(axis=2) - a.min(axis=2)
    e = gx + gy + 0.25 * sat
    return np.asarray(Image.fromarray((e / (e.max() + 1e-6) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(3)), dtype=np.float32)


def subject_box(im: Image.Image, frac: float = 0.6) -> tuple[float, float, float, float]:
    """(x, y, w, h) in 0..1 of the frac-sized 16:9 window with the most detail."""
    e = energy(im)
    h, w = e.shape
    bw, bh = int(w * frac), int(h * frac)
    ii = np.pad(e.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    sums = ii[bh:, bw:] - ii[:-bh, bw:] - ii[bh:, :-bw] + ii[:-bh, :-bw]
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return x / w, y / h, bw / w, bh / h


def subject_side(im: Image.Image) -> str:
    """'left' or 'right': where most of the detail sits."""
    e = energy(im)
    half = e.shape[1] // 2
    return "left" if e[:, :half].sum() >= e[:, half:].sum() else "right"


def side_gradient(im: Image.Image, side: str, reach: float = 0.62, strength: float = 0.88) -> Image.Image:
    """Darken one side with a smooth gradient reaching `reach` of the width."""
    w, h = im.size
    x = np.linspace(0, 1, w, dtype=np.float32)
    ramp = np.clip(1 - x / reach, 0, 1) if side == "left" else np.clip((x - (1 - reach)) / reach, 0, 1)
    alpha = (strength * ramp ** 1.3 * 255).astype(np.uint8)
    mask = Image.fromarray(np.tile(alpha, (h, 1)), "L")
    return Image.composite(Image.new("RGB", im.size, (0, 0, 0)), im, mask)


def vignette(im: Image.Image, strength: float = 0.55) -> Image.Image:
    w, h = im.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2.4
    mask = Image.fromarray((np.clip(d, 0, 1) * strength * 255).astype(np.uint8), "L")
    return Image.composite(Image.new("RGB", im.size, (0, 0, 0)), im, mask)


# ---------------------------------------------------------------- compositions

def comp_gradient(src: Image.Image, text: str, font: Path) -> Image.Image:
    """Composition 1: graded frame, dark gradient on the text side, huge two-colour text."""
    im = punch(cover(src))
    text_side = "right" if subject_side(im) == "left" else "left"
    im = side_gradient(im, text_side)
    max_w, max_h = int(SIZE[0] * 0.54), int(SIZE[1] * 0.72)
    lines, f = fit_text(text, font, max_w, max_h)
    block_h = int(f.size * 1.02 * len(lines))
    block_w = int(max(f.getlength(ln) for ln in lines))
    margin = 56
    x = margin if text_side == "left" else SIZE[0] - margin - block_w
    draw_text(im, lines, f, x, (SIZE[1] - block_h) // 2, [(255, 255, 255), ACCENT])
    return im


def comp_subject(src: Image.Image, text: str, font: Path) -> Image.Image:
    """Composition 2: zoom to the subject, yellow block with black text, red arrow toward the subject."""
    bx, by, bw, bh = subject_box(src)
    w, h = src.size
    crop = src.crop((int(bx * w), int(by * h), int((bx + bw) * w), int((by + bh) * h)))
    im = vignette(punch(cover(crop), 1.18, 1.3), 0.35)
    lines, f = fit_text(text, font, int(SIZE[0] * 0.5), int(SIZE[1] * 0.36), top=150)
    pad = int(f.size * 0.28)
    block_w = int(max(f.getlength(ln) for ln in lines)) + 2 * pad
    block_h = int(f.size * 1.02 * len(lines)) + 2 * pad
    x0, y1 = 40, SIZE[1] - 40
    y0 = y1 - block_h
    d = ImageDraw.Draw(im)
    d.polygon([(x0, y0), (x0 + block_w + 24, y0), (x0 + block_w, y1), (x0 - 12, y1)], fill=ACCENT,
              outline=(0, 0, 0), width=6)
    for i, ln in enumerate(lines):
        d.text((x0 + pad, y0 + pad * 0.6 + i * f.size * 1.02), ln, font=f, fill=(15, 15, 15))
    start = np.array([x0 + block_w + 36, y0 + 24], dtype=np.float32)
    target = np.array([SIZE[0] * 0.6, SIZE[1] * 0.38], dtype=np.float32)
    if target[0] - start[0] < 160:
        target = start + np.array([240, -190], dtype=np.float32)
    target = np.minimum(target, np.array([SIZE[0] - 60, SIZE[1] - 60], dtype=np.float32))
    _arrow(im, (int(start[0]), int(start[1])), (int(target[0]), int(target[1])))
    return im


def _arrow(im: Image.Image, start: tuple[int, int], end: tuple[int, int], width: int = 26) -> None:
    """Thick red arrow with a white outline, from start toward end."""
    d = ImageDraw.Draw(im)
    fill = _contrasting(im, start, end, [ARROW, ACCENT])
    v = np.array(end, dtype=np.float32) - np.array(start, dtype=np.float32)
    length = float(np.linalg.norm(v)) or 1.0
    u = v / length
    n = np.array([-u[1], u[0]])
    head = width * 2.6
    base = np.array(end) - u * head
    for colour, grow in (((255, 255, 255), 6), (fill, 0)):
        d.line([start, tuple(base)], fill=colour, width=width + 2 * grow)
        tip = np.array(end) + u * grow
        pts = [tuple(tip), tuple(base + n * (head * 0.75 + grow)), tuple(base - n * (head * 0.75 + grow))]
        d.polygon(pts, fill=colour)


def _contrasting(im: Image.Image, start: tuple[int, int], end: tuple[int, int],
                 options: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """The option farthest in RGB from the mean colour of the box spanned by start and end."""
    x0, x1 = sorted((start[0], end[0]))
    y0, y1 = sorted((start[1], end[1]))
    region = np.asarray(im.crop((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1))), dtype=np.float32)
    mean = region.reshape(-1, 3).mean(axis=0)
    return max(options, key=lambda c: float(np.linalg.norm(np.array(c, dtype=np.float32) - mean)))


def comp_centered(src: Image.Image, text: str, font: Path) -> Image.Image:
    """Composition 3: AI background, vignette, centred text over an accent bar."""
    im = vignette(punch(cover(src), 1.15, 1.25), 0.7)
    lines, f = fit_text(text, font, int(SIZE[0] * 0.86), int(SIZE[1] * 0.5))
    block_w = int(max(f.getlength(ln) for ln in lines))
    block_h = int(f.size * 1.02 * len(lines))
    x = (SIZE[0] - block_w) // 2
    y = int(SIZE[1] * 0.92) - block_h - 28
    shade = Image.new("L", SIZE, 0)
    ImageDraw.Draw(shade).rectangle([0, y - 40, SIZE[0], SIZE[1]], fill=150)
    im.paste(Image.new("RGB", SIZE, (0, 0, 0)), (0, 0), shade.filter(ImageFilter.GaussianBlur(40)))
    draw_text(im, lines, f, x, y, [(255, 255, 255)], align="center")
    ImageDraw.Draw(im).rectangle([x, y + block_h + 8, x + block_w, y + block_h + 22], fill=ACCENT)
    return im
