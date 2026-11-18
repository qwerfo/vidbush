"""Numbered contact sheets of candidate keyframes for the edit call.

The vision seat downsizes every image to 640 px on its longest side, so a sheet is 640 px wide and holds
ROWS candidates, one per row: a video shows its three keyframes a, b, c side by side, a photograph its one
frame a. Each row is labelled with the candidate number (and AI for a generated still).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

W = 640
ROWS = 4
FRAME_W, FRAME_H = 212, 119
CAP_H = 24
ROW_H = FRAME_H + CAP_H
LETTERS = "abc"


def _font(size: int):
    for name in ("arialbd.ttf", "arial.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _fit(path: str, w: int, h: int) -> Image.Image:
    tile = Image.new("RGB", (w, h), (24, 24, 24))
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((w, h))
            tile.paste(im, ((w - im.width) // 2, (h - im.height) // 2))
    except OSError:
        pass
    return tile


def build(entries: list[dict], out_dir: Path, prefix: str) -> list[str]:
    """entries: {n, keyframes, ai, media}; returns sheet paths, ROWS candidates per sheet."""
    out_dir.mkdir(parents=True, exist_ok=True)
    big, small = _font(18), _font(14)
    paths = []
    for s in range(0, len(entries), ROWS):
        chunk = entries[s:s + ROWS]
        sheet = Image.new("RGB", (W, ROW_H * len(chunk)), (0, 0, 0))
        d = ImageDraw.Draw(sheet)
        for r, e in enumerate(chunk):
            y = r * ROW_H
            label = f"#{e['n']}" + ("  AI generated" if e.get("ai") else "") + \
                    ("  video" if e.get("media") == "video" else "  photo")
            d.rectangle([0, y, W, y + CAP_H - 1], fill=(250, 210, 40) if not e.get("ai") else (230, 90, 90))
            d.text((6, y + 2), label, fill=(0, 0, 0), font=big)
            frames = list(e.get("keyframes") or [])[:3]
            if e.get("media") == "video" and len(frames) >= 2:
                for k, f in enumerate(frames):
                    x = k * (FRAME_W + 2)
                    sheet.paste(_fit(f, FRAME_W, FRAME_H), (x, y + CAP_H))
                    d.rectangle([x, y + CAP_H, x + 18, y + CAP_H + 18], fill=(0, 0, 0))
                    d.text((x + 4, y + CAP_H), LETTERS[k], fill=(255, 255, 255), font=small)
            elif frames:
                sheet.paste(_fit(frames[0], W, FRAME_H), (0, y + CAP_H))
        p = out_dir / f"{prefix}_{s // ROWS + 1:02d}.jpg"
        sheet.save(p, quality=88)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------- picker cards (one asset per image)

CARD_W = 640  # the vision seat fits every image into 640 px; one asset per image keeps it at full width
CARD_MAIN_H = 360
STRIP_H = 120
CARD_MAX_H = 640


def asset_card(n: int, rec: dict, out_dir: Path, prefix: str) -> Optional[str]:
    """One numbered image for one asset: a still at full width (documents keep their height up to 640 px),
    a video's first frame large with its other frames below, each with its time in the asset."""
    frames = [(f["t"], f["path"]) for f in rec.get("frames", []) if Path(f["path"]).exists()]
    if not frames:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    big, small = _font(18), _font(13)
    video = rec.get("media") == "video"
    if video:
        h = CAP_H + CARD_MAIN_H + (STRIP_H if len(frames) > 1 else 0)
        card = Image.new("RGB", (CARD_W, h), (0, 0, 0))
        card.paste(_fit(frames[0][1], CARD_W, CARD_MAIN_H), (0, CAP_H))
        d = ImageDraw.Draw(card)
        d.rectangle([0, CAP_H, 64, CAP_H + 18], fill=(0, 0, 0))
        d.text((4, CAP_H + 1), f"{frames[0][0]:.1f}s", fill=(255, 255, 255), font=small)
        rest = frames[1:4]
        w = CARD_W // max(1, len(rest)) if rest else CARD_W
        for k, (t, p) in enumerate(rest):
            x = k * w
            card.paste(_fit(p, w - 2, STRIP_H - 2), (x, CAP_H + CARD_MAIN_H + 2))
            d.rectangle([x, CAP_H + CARD_MAIN_H + 2, x + 60, CAP_H + CARD_MAIN_H + 20], fill=(0, 0, 0))
            d.text((x + 4, CAP_H + CARD_MAIN_H + 3), f"{t:.1f}s", fill=(255, 255, 255), font=small)
    else:
        with Image.open(frames[0][1]) as im:
            iw, ih = im.size
        body_h = max(120, min(CARD_MAX_H - CAP_H, round(CARD_W * ih / max(1, iw))))
        card = Image.new("RGB", (CARD_W, CAP_H + body_h), (0, 0, 0))
        card.paste(_fit(frames[0][1], CARD_W, body_h), (0, CAP_H))
        d = ImageDraw.Draw(card)
    ai = rec.get("ai")
    d.rectangle([0, 0, CARD_W, CAP_H - 1], fill=(230, 90, 90) if ai else (250, 210, 40))
    d.text((6, 2), f"#{n}" + ("  AI generated" if ai else "") + ("  video" if video else "  still"),
           fill=(0, 0, 0), font=big)
    p = out_dir / f"{prefix}_a{n:03d}.jpg"
    card.save(p, quality=90)
    return str(p)
