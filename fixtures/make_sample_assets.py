"""Generate the bundled sample asset library (fixtures/sample_assets).

A synthetic CC0 local footage library so the offline demo has moving footage
and stills to work with. Every clip and image is visibly labelled
"SAMPLE FOOTAGE <n>" or "SAMPLE IMAGE <n>" so nobody mistakes it for real
footage. Each file gets a sidecar <stem>.json with title, tags, licence,
author, era and description, used by vidbush/source/local.py.

Run: python fixtures/make_sample_assets.py [--only-stills] [--only-video]
Deterministic: the same files come out on every run.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures" / "sample_assets"
FONT = ROOT / "assets" / "fonts" / "Inter-Bold.otf"
LICENCE = "CC0 (Vidbush generated fixture)"
AUTHOR = "Vidbush fixtures"

# ---------------------------------------------------------------- video looks

SEPIA = "colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131"
LOOKS: dict[str, str] = {
    "sepia": f"format=rgb24,{SEPIA},noise=alls=9:allf=t,vignette=PI/4.5",
    "mono": "format=rgb24,noise=alls=10:allf=t,vignette=PI/4.5",
    "teal": "format=rgb24,lutrgb=r='12+val*0.55':g='28+val*0.78':b='40+val*0.86',noise=alls=4:allf=t",
    "amber": "format=rgb24,lutrgb=r='30+val*0.88':g='18+val*0.66':b='10+val*0.42',noise=alls=4:allf=t",
    "steel": "format=rgb24,lutrgb=r='18+val*0.70':g='22+val*0.76':b='30+val*0.85',noise=alls=4:allf=t",
    "dusk": "format=rgb24,lutrgb=r='40+val*0.82':g='14+val*0.62':b='45+val*0.66',noise=alls=4:allf=t",
}
# archival flicker and a wandering scratch, multiplied into gray geq expressions
FLICKER = "(1+0.05*sin(T*31)*sin(T*7))"
SCRATCH = "-70*lt(abs(X-W*(0.5+0.4*sin(T*1.3))),0.8)"

# gray geq expressions on a 640x360 canvas, every one animated by T
SCENES: dict[str, str] = {
    "hills": ("if(gt(Y,H*0.72+22*sin((X+T*42)/80+1)+9*sin((X+T*42)/27)),38,"
              "if(gt(Y,H*0.6+20*sin((X+T*24)/57)+10*sin((X+T*24)/21)),82,"
              "if(gt(Y,H*0.5+16*sin((X+T*12)/66+2)+6*sin((X+T*12)/19)),128,215-Y*0.22)))"),
    "sea_ship": ("if(between(X,W*0.1+T*11,W*0.1+T*11+120)*between(Y,H*0.36,H*0.43)"
                 "+between(X,W*0.1+T*11+40,W*0.1+T*11+80)*between(Y,H*0.3,H*0.36)"
                 "+between(X,W*0.1+T*11+52,W*0.1+T*11+64)*between(Y,H*0.2,H*0.3),24,"
                 "if(lt(Y,H*0.42),225-Y*0.35,"
                 "70+(Y/H)*90*(0.5+0.25*sin(X/17+T*1.9+2*sin(Y/29-T*0.6))+0.25*sin(X/7-T*2.7))))"),
    "gears": ("if(lt(hypot(X-W*0.38,Y-H*0.5),95+12*gt(sin(14*(atan2(Y-H*0.5,X-W*0.38)+T*0.6)),0))"
              "*gt(hypot(X-W*0.38,Y-H*0.5),28),62,"
              "if(lt(hypot(X-W*0.68,Y-H*0.42),62+10*gt(sin(10*(atan2(Y-H*0.42,X-W*0.68)-T*0.84)),0))"
              "*gt(hypot(X-W*0.68,Y-H*0.42),18),88,150+40*Y/H))"),
    "skyline": ("if(gt(Y,H-(H*0.25+H*0.42*mod(abs(sin(floor((X+T*14)/34)*12.9898))*437.585,1))),"
                "if(mod(floor((X+T*14)/6),2)*mod(floor(Y/9),2)*gt(sin(floor((X+T*14)/6)*7.1+floor(Y/9)*3.3+floor(T*2)),0.3),"
                "200,30),90+Y*0.35)"),
    "light_bars": ("20+200*pow(0.5+0.5*sin((X+Y*0.4)/23-T*2.3),12)"
                   "+60*pow(0.5+0.5*sin(X/71+T*0.7),6)"),
    "contours": ("200-110*lt(abs(sin(6*(sin((X+T*9)/41)+cos(Y/37)+sin((X+T*9+Y)/59)))),0.09)"
                 "-30*lt(mod(X+T*9,80),1.2)-30*lt(mod(Y,80),1.2)"),
    "workers": ("if(lt(hypot(mod(X-T*20,34)-17,Y-H*0.5),6)+between(mod(X-T*20,34),11,23)*between(Y,H*0.5+6,H*0.5+44)"
                "+lt(hypot(mod(X-T*30,52)-26,Y-H*0.68),9)+between(mod(X-T*30,52),17,35)*between(Y,H*0.68+9,H*0.68+70),"
                "30,if(gt(Y,H*0.8),95,190-Y*0.2))"),
    "clouds": ("128+38*sin(X/53+T*0.3)*cos(Y/41-T*0.2)+28*sin((X+Y)/29+T*0.5)"
               "+18*sin((X-Y*2)/17-T*0.4)+12*cos(X/11+Y/13+T*0.9)"),
    "rails": ("if(gt(Y,H*0.42)*lt(abs(X-W/2),(Y-H*0.42)*1.6),"
              "if(lt(abs(abs(X-W/2)-(Y-H*0.42)*0.9),1+(Y-H*0.42)*0.03),220,"
              "if(lt(mod(900/(Y-H*0.42+4)+T*3,1),0.35),70,120)),if(gt(Y,H*0.42),60,200-Y*0.3))"),
    "canal_cut": ("if(lt(Y,H*0.28),215-Y*0.4,"
                  "if(lt(abs(X-W/2),30+34*floor((H-Y)/28)),"
                  "if(gt(Y,H*0.86-T*3),105+18*sin(X/9+T*2)*sin(Y/5-T*3),150+22*mod(floor(Y/28),2)),"
                  "70+10*sin(X/13+Y/17)))"),
}


@dataclass
class Clip:
    n: int
    scene: str  # key in SCENES, or a native lavfi source: mandelbrot, life, cellauto, gradients
    look: str
    size: tuple[int, int]
    seconds: float
    title: str
    tags: list[str]
    era: str
    description: str
    archival: bool = False
    extra: dict = field(default_factory=dict)


def clip_specs() -> list[Clip]:
    """The 24 sample clips: varied motion, palette and size, topical tags."""
    hd, sd = (1920, 1080), (1280, 720)
    c = Clip
    return [
        c(1, "sea_ship", "sepia", hd, 12, "Steamship crossing open water", ["ships", "water", "steamship", "ocean", "archival", "history", "voyage"], "1900s", "A small steamer moves left to right across a choppy sea, sepia archival look.", True),
        c(2, "sea_ship", "teal", sd, 10, "Cargo ship on calm sea", ["ships", "water", "ocean", "cargo", "trade", "shipping"], "modern", "A ship silhouette sails across moving waves under a pale sky."),
        c(3, "hills", "sepia", hd, 13, "Jungle hills panorama", ["landscape", "jungle", "hills", "isthmus", "archival", "history", "terrain"], "1900s", "Layered hills drift past in parallax, sepia film texture.", True),
        c(4, "hills", "dusk", sd, 11, "Mountain ridges at dusk", ["landscape", "mountains", "hills", "dusk", "terrain"], "modern", "Coloured ridges slide by at different speeds at dusk."),
        c(5, "gears", "mono", sd, 9, "Machinery gears turning", ["machines", "machinery", "gears", "engineering", "industrial", "archival"], "1910s", "Two interlocking gears rotate in opposite directions, black and white.", True),
        c(6, "gears", "steel", hd, 10, "Lock gate machinery", ["machines", "lock", "gate", "engineering", "construction", "canal"], "1910s", "Large toothed wheels turn slowly, cold steel tint."),
        c(7, "skyline", "amber", sd, 12, "City skyline with lit windows", ["city", "skyline", "buildings", "night", "urban", "port city"], "modern", "A city skyline pans sideways while windows flicker on and off."),
        c(8, "skyline", "mono", hd, 10, "Old harbour town rooftops", ["city", "town", "buildings", "history", "archival", "harbour"], "1900s", "Rooftops of a harbour town pass by, black and white archival look.", True),
        c(9, "light_bars", "amber", sd, 8, "Welding light in the workshop", ["machines", "workers", "construction", "industrial", "light", "workshop"], "modern", "Moving bars of light sweep across a dark industrial space."),
        c(10, "light_bars", "steel", sd, 9, "Industrial light beams", ["machines", "industrial", "factory", "light", "steel"], "modern", "Cold light beams travel diagonally across the frame."),
        c(11, "contours", "sepia", hd, 12, "Survey map contours", ["map", "survey", "terrain", "isthmus", "history", "archival", "route"], "1880s", "Topographic contour lines drift slowly across a parchment surface.", True),
        c(12, "contours", "teal", sd, 10, "Topographic map pan", ["map", "terrain", "landscape", "route", "geography"], "modern", "A coloured topographic pattern pans over a grid."),
        c(13, "workers", "sepia", hd, 11, "Workers marching to the dig", ["workers", "labourers", "construction", "history", "archival", "crowd"], "1900s", "Rows of workers walk across the frame toward the excavation, sepia film.", True),
        c(14, "workers", "mono", sd, 10, "Crowd of labourers", ["workers", "crowd", "labourers", "people", "archival", "history"], "1910s", "Two rows of figures move at different speeds, black and white.", True),
        c(15, "clouds", "teal", sd, 12, "Drifting tropical clouds", ["sky", "clouds", "weather", "rain", "landscape", "tropics"], "modern", "Soft cloud forms drift and morph slowly."),
        c(16, "clouds", "sepia", sd, 9, "Storm clouds archival", ["sky", "storm", "weather", "clouds", "archival", "history"], "1900s", "Turbulent cloud textures roll past, sepia archival look.", True),
        c(17, "rails", "mono", hd, 10, "Railway tracks forward motion", ["railway", "railroad", "train", "construction", "history", "archival", "tracks"], "1850s", "A point of view down railway tracks with sleepers rushing toward the camera.", True),
        c(18, "rails", "amber", sd, 9, "Rail line at sunset", ["railway", "railroad", "tracks", "transport", "landscape"], "modern", "Rail sleepers stream toward the camera under a warm sky."),
        c(19, "canal_cut", "sepia", hd, 14, "Excavated cut filling with water", ["construction", "canal", "excavation", "water", "cut", "history", "archival"], "1910s", "A stepped excavation through hills while water slowly rises at the bottom.", True),
        c(20, "canal_cut", "teal", sd, 12, "Canal channel and terraces", ["canal", "water", "construction", "landscape", "channel"], "modern", "A stepped channel with rising rippling water."),
        c(21, "mandelbrot", "amber", sd, 10, "Abstract fractal zoom", ["abstract", "texture", "pattern", "transition"], "modern", "A slow colourful fractal zoom, abstract background texture.",
          extra={"start_x": -0.743643887037151, "start_y": 0.13182590420533}),
        c(22, "life", "teal", sd, 9, "Aerial lights pattern", ["city", "lights", "aerial", "night", "map", "abstract"], "modern", "A living pattern of small lights spreading and fading, like a city seen from above."),
        c(23, "cellauto", "sepia", sd, 8, "Loom pattern machinery", ["machines", "textile", "industrial", "pattern", "archival"], "1900s", "A scrolling woven pattern reminiscent of an industrial loom, sepia tone.", True),
        c(24, "gradients", "dusk", hd, 12, "Dawn over the water", ["sky", "dawn", "water", "landscape", "sunrise", "ocean"], "modern", "Slowly shifting dawn colours with gentle movement."),
    ]


def _source(clip: Clip) -> str:
    """Filtergraph source chain producing an animated frame for a clip."""
    d = clip.seconds
    if clip.scene in SCENES:
        expr = SCENES[clip.scene]
        if clip.archival:
            expr = f"({expr})*{FLICKER}{SCRATCH}"
        return f"color=c=black:s=640x360:r=30:d={d},format=gray,geq=lum='clip({expr},0,255)'"
    if clip.scene == "mandelbrot":
        x, y = clip.extra["start_x"], clip.extra["start_y"]
        return (f"mandelbrot=s=640x360:r=30:maxiter=200:start_x={x}:start_y={y}:start_scale=2.5:"
                f"end_scale=0.05:end_pts={int(d * 30)}:outer=normalized_iteration_count,trim=duration={d},format=gray")
    if clip.scene == "life":
        return (f"life=s=320x180:r=30:mold=20:ratio=0.12:seed={clip.n}:life_color=white:"
                f"death_color=black:mold_color=0x404040,trim=duration={d},format=gray,gblur=sigma=0.8")
    if clip.scene == "cellauto":
        return f"cellauto=s=320x180:r=30:rule=90:seed={clip.n},trim=duration={d},format=gray,gblur=sigma=0.7"
    return (f"gradients=s=640x360:r=30:c0=0x0b1a33:c1=0x8a4a3a:c2=0xe2a65a:nb_colors=3:speed=0.012:"
            f"seed={clip.n}:duration={d},format=gray")


def _label(text: str, w: int, h: int) -> str:
    size = int(h * 0.03)
    return (f"drawtext=fontfile='{FONT}':text='{text}':fontsize={size}:fontcolor=white@0.9:"
            f"box=1:boxcolor=black@0.5:boxborderw={size // 3}:x=w*0.03:y=h*0.145")


def render_clip(clip: Clip, out: Path) -> None:
    """Render one labelled, silent H.264 clip with ffmpeg lavfi."""
    w, h = clip.size
    look = LOOKS[clip.look]
    if clip.scene == "gradients":
        look = "format=rgb24,noise=alls=4:allf=t"
    chain = (f"{_source(clip)},{look},scale={w}:{h}:flags=bicubic,"
             f"{_label(f'SAMPLE FOOTAGE {clip.n:02d}', w, h)},format=yuv420p")
    if clip.scene == "gradients":
        chain = chain.replace(",format=gray", "", 1)
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", chain,
           "-t", str(clip.seconds), "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27",
           "-maxrate", "1100k" if w > 1280 else "850k", "-bufsize", "2000k", "-g", "30", "-r", "30", "-movflags", "+faststart",
           str(out)]
    subprocess.run(cmd, check=True)


def write_sidecar(media: Path, title: str, tags: list[str], era: str, description: str, **extra: object) -> None:
    data = {"title": title, "tags": tags, "licence": LICENCE, "author": AUTHOR, "source_url": "",
            "era": era, "description": description, **extra}
    media.with_suffix(".json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def make_videos() -> None:
    for clip in clip_specs():
        out = OUT / f"sample_footage_{clip.n:02d}.mp4"
        render_clip(clip, out)
        write_sidecar(out, clip.title, clip.tags, clip.era, clip.description,
                      duration=clip.seconds, width=clip.size[0], height=clip.size[1])
        print(f"{out.name} {out.stat().st_size // 1024} KB", flush=True)


# ---------------------------------------------------------------- stills

def _ridge(rng: np.random.Generator, w: int, base: float, rough: float, octaves: int = 7) -> np.ndarray:
    """1D fractal ridge line, heights in pixels."""
    x = np.linspace(0, 1, w)
    y = np.full(w, base)
    amp = rough
    for o in range(octaves):
        freq = 2 ** o * 1.5
        phase = rng.uniform(0, 2 * np.pi)
        y += amp * np.sin(2 * np.pi * freq * x + phase) * rng.uniform(0.6, 1.0)
        amp *= 0.52
    return y


def _noise2d(rng: np.random.Generator, w: int, h: int, octaves: int = 6) -> np.ndarray:
    """Smooth value noise in 0..1 built from upsampled random grids."""
    acc = np.zeros((h, w), np.float32)
    total = 0.0
    for o in range(octaves):
        gw, gh = 4 * 2 ** o, max(2, int(4 * 2 ** o * h / w))
        grid = (rng.random((gh, gw)) * 255).astype(np.uint8)
        up = np.asarray(Image.fromarray(grid).resize((w, h), Image.BICUBIC), np.float32) / 255.0
        weight = 0.55 ** o
        acc += up * weight
        total += weight
    acc /= total
    return (acc - acc.min()) / max(1e-6, acc.max() - acc.min())


def _sky(w: int, h: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    t = np.linspace(0, 1, h)[:, None, None]
    return (np.array(top)[None, None, :] * (1 - t) + np.array(bottom)[None, None, :] * t).repeat(w, 1).astype(np.float32)


def still_landscape(rng: np.random.Generator, w: int, h: int, top: tuple, bottom: tuple, ridge: tuple) -> Image.Image:
    """Layered ridges with atmospheric perspective under a gradient sky and sun."""
    img = _sky(w, h, top, bottom)
    yy, xx = np.mgrid[0:h, 0:w]
    sun = np.exp(-(((xx - w * rng.uniform(0.3, 0.7)) ** 2 + (yy - h * 0.38) ** 2) / (2 * (h * 0.05) ** 2)))
    img += sun[..., None] * np.array([90, 70, 40])
    layers = 5
    for i in range(layers):
        depth = i / (layers - 1)
        line = _ridge(rng, w, h * (0.45 + 0.11 * i), h * (0.09 - 0.012 * i))
        mask = yy > line[None, :]
        col = np.array(ridge) * (0.35 + 0.65 * (1 - depth)) + np.array(bottom) * (1 - depth) * 0.5
        col = col * (1 - 0.65 * depth) + np.array(bottom) * 0.25 * (1 - depth)
        img[mask] = col
    fog = _noise2d(rng, w, h, 5)[..., None]
    img = img * (0.92 + 0.08 * fog)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def still_clouds(rng: np.random.Generator, w: int, h: int, top: tuple, bottom: tuple) -> Image.Image:
    img = _sky(w, h, top, bottom)
    n = _noise2d(rng, w, h, 7)
    cloud = np.clip((n - 0.45) * 2.6, 0, 1)[..., None]
    img = img * (1 - cloud * 0.85) + np.array([245, 240, 232]) * cloud * 0.85
    shade = np.clip((n - 0.6) * 3, 0, 1)[..., None]
    img = img * (1 - shade * 0.25)
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def _draw_industrial(d: ImageDraw.ImageDraw, rng: np.random.Generator, w: int, h: int, kind: str, ink: tuple) -> None:
    ground = int(h * 0.78)
    d.rectangle([0, ground, w, h], fill=ink)
    if kind == "lock":
        for x0 in (int(w * 0.18), int(w * 0.52)):
            d.rectangle([x0, int(h * 0.42), x0 + int(w * 0.28), ground], fill=ink)
            for k in range(6):
                y = int(h * 0.45) + k * int(h * 0.055)
                d.line([x0 + 10, y, x0 + int(w * 0.28) - 10, y], fill=(200, 190, 170), width=3)
        d.rectangle([0, int(h * 0.7), w, ground], fill=tuple(int(c * 1.8) for c in ink))
    elif kind == "truss":
        y0, y1 = int(h * 0.42), int(h * 0.55)
        d.rectangle([0, y1, w, y1 + 14], fill=ink)
        d.line([0, y0, w, y0], fill=ink, width=12)
        step = w // 24
        for i in range(25):
            x = i * step
            d.line([x, y0, x + step, y1], fill=ink, width=8)
            d.line([x + step, y0, x, y1], fill=ink, width=8)
        for x in range(step * 3, w, step * 6):
            d.rectangle([x - 18, y1, x + 18, ground], fill=ink)
    elif kind == "cranes":
        for i in range(3):
            x = int(w * (0.15 + 0.3 * i + rng.uniform(-0.04, 0.04)))
            top = int(h * rng.uniform(0.18, 0.3))
            d.rectangle([x - 14, top, x + 14, ground], fill=ink)
            d.line([x - int(w * 0.08), top + 20, x + int(w * 0.18), top + 20], fill=ink, width=16)
            hook_x = x + int(w * rng.uniform(0.06, 0.16))
            d.line([hook_x, top + 20, hook_x, int(h * 0.55)], fill=ink, width=3)
            d.rectangle([hook_x - 30, int(h * 0.55), hook_x + 30, int(h * 0.6)], fill=ink)
    else:  # stacks
        for i in range(7):
            x = int(w * (0.08 + 0.13 * i))
            top = int(h * rng.uniform(0.2, 0.45))
            d.rectangle([x - 22, top, x + 22, ground], fill=ink)
        d.rectangle([int(w * 0.05), int(h * 0.55), int(w * 0.95), ground], fill=ink)


def still_industrial(rng: np.random.Generator, w: int, h: int, kind: str, top: tuple, bottom: tuple) -> Image.Image:
    img = still_clouds(rng, w, h, top, bottom)
    d = ImageDraw.Draw(img)
    _draw_industrial(d, rng, w, h, kind, (28, 26, 30))
    if kind == "stacks":
        smoke = _noise2d(rng, w, h, 6)
        arr = np.asarray(img, np.float32)
        mask = np.clip((smoke - 0.55) * 2.5, 0, 1) * (np.linspace(1, 0, h)[:, None] ** 2)
        arr = arr * (1 - mask[..., None] * 0.6) + 90 * mask[..., None] * 0.6
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def still_ship(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    img = still_clouds(rng, w, h, (180, 180, 175), (220, 215, 205))
    arr = np.asarray(img, np.float32)
    horizon = int(h * 0.58)
    waves = _noise2d(rng, w, h - horizon, 6)
    arr[horizon:] = (60 + 70 * waves)[..., None] * np.array([1.0, 1.0, 1.0])
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    x0 = int(w * rng.uniform(0.2, 0.45))
    hull = [(x0, horizon - 40), (x0 + int(w * 0.42), horizon - 40), (x0 + int(w * 0.39), horizon + 25), (x0 + 30, horizon + 25)]
    d.polygon(hull, fill=(25, 24, 22))
    d.rectangle([x0 + int(w * 0.12), horizon - 110, x0 + int(w * 0.28), horizon - 40], fill=(32, 30, 28))
    for k in range(2):
        fx = x0 + int(w * (0.16 + 0.07 * k))
        d.rectangle([fx, horizon - 210, fx + 34, horizon - 110], fill=(25, 24, 22))
    d.line([x0 + int(w * 0.05), horizon - 40, x0 + int(w * 0.08), horizon - 260], fill=(25, 24, 22), width=6)
    d.line([x0 + int(w * 0.36), horizon - 40, x0 + int(w * 0.33), horizon - 240], fill=(25, 24, 22), width=6)
    return img.filter(ImageFilter.GaussianBlur(1.0))


def still_map(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    n = _noise2d(rng, w, h, 7)
    yy = np.linspace(-1, 1, h)[:, None]
    land = (n + 0.35 * (1 - np.abs(yy * 1.4))) > 0.62
    paper = np.array([226, 210, 176], np.float32)
    sea = np.array([182, 196, 190], np.float32)
    img = np.where(land[..., None], paper, sea)
    contour = np.abs(np.sin(n * 40)) < 0.06
    img[contour & land] *= 0.78
    img[(np.arange(w)[None, :] % 160 < 2) | (np.arange(h)[:, None] % 160 < 2)] *= 0.85
    out = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    edge = out.convert("L").filter(ImageFilter.FIND_EDGES).point(lambda v: 255 if v > 18 else 0)
    out.paste((90, 70, 50), mask=edge)
    return out


def still_night_city(rng: np.random.Generator, w: int, h: int) -> Image.Image:
    img = Image.fromarray(_sky(w, h, (8, 12, 30), (40, 34, 60)).astype(np.uint8))
    d = ImageDraw.Draw(img)
    x = 0
    while x < w:
        bw = int(rng.uniform(60, 180))
        top = int(h * rng.uniform(0.3, 0.75))
        d.rectangle([x, top, x + bw, h], fill=(14, 14, 20))
        for wy in range(top + 12, h - 10, 22):
            for wx in range(x + 8, x + bw - 10, 18):
                if rng.random() < 0.35:
                    d.rectangle([wx, wy, wx + 8, wy + 10], fill=(250, 210, 120))
        x += bw + int(rng.uniform(4, 20))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def age_photo(img: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Sepia tone, grain, scratches, vignette and a faded border like an old print."""
    g = np.asarray(img.convert("L"), np.float32)
    h, w = g.shape
    g += rng.normal(0, 9, g.shape)
    for _ in range(14):
        x = int(rng.uniform(0, w))
        g[:, max(0, x - 1):x + 1] += rng.uniform(-50, 40)
    yy, xx = np.mgrid[0:h, 0:w]
    vig = 1 - 0.55 * (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) ** 1.5
    g = np.clip(g * np.clip(vig, 0.2, 1), 0, 255) / 255.0
    sep = np.stack([g * 240 + 15, g * 205 + 12, g * 160 + 8], -1)
    return Image.fromarray(np.clip(sep, 0, 255).astype(np.uint8))


def label_image(img: Image.Image, text: str) -> Image.Image:
    w, h = img.size
    d = ImageDraw.Draw(img, "RGBA")
    size = int(h * 0.03)
    font = ImageFont.truetype(str(FONT), size)
    x, y = int(w * 0.03), int(h * 0.145)
    box = d.textbbox((x, y), text, font=font)
    pad = size // 3
    d.rectangle([box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad], fill=(0, 0, 0, 128))
    d.text((x, y), text, font=font, fill=(255, 255, 255, 230))
    return img


def still_specs() -> list[tuple[int, str, tuple[int, int], dict]]:
    """(n, kind, size, sidecar fields) for the 16 sample stills."""
    big, hd = (2400, 1350), (1920, 1080)
    return [
        (1, "landscape_dawn", big, {"title": "Mountain ridges at dawn", "tags": ["landscape", "mountains", "dawn", "hills", "terrain"], "era": "modern"}),
        (2, "landscape_dusk", hd, {"title": "Jungle hills at dusk", "tags": ["landscape", "jungle", "hills", "isthmus", "dusk"], "era": "modern"}),
        (3, "landscape_sepia", big, {"title": "Old photograph of rolling hills", "tags": ["landscape", "hills", "archival", "history", "photograph", "isthmus"], "era": "1900s"}),
        (4, "landscape_blue", hd, {"title": "Misty valley", "tags": ["landscape", "valley", "mist", "river", "water"], "era": "modern"}),
        (5, "clouds_day", hd, {"title": "Tropical sky with clouds", "tags": ["sky", "clouds", "weather", "tropics", "landscape"], "era": "modern"}),
        (6, "clouds_storm", big, {"title": "Storm clouds over the sea", "tags": ["sky", "storm", "weather", "clouds", "water", "ocean"], "era": "modern"}),
        (7, "lock", big, {"title": "Canal lock gates", "tags": ["canal", "lock", "gate", "construction", "machines", "water", "engineering"], "era": "1910s"}),
        (8, "truss", hd, {"title": "Steel truss bridge", "tags": ["construction", "bridge", "steel", "engineering", "railway"], "era": "1900s"}),
        (9, "cranes", big, {"title": "Cranes over the construction site", "tags": ["construction", "cranes", "machines", "workers", "industrial", "excavation"], "era": "1910s"}),
        (10, "stacks", hd, {"title": "Smokestacks of an industrial town", "tags": ["industrial", "factory", "machines", "city", "smoke", "history"], "era": "1900s"}),
        (11, "ship_sepia", big, {"title": "Old photograph of a steamship", "tags": ["ships", "steamship", "water", "ocean", "archival", "history", "photograph"], "era": "1900s"}),
        (12, "ship_mono", hd, {"title": "Steamer at anchor", "tags": ["ships", "steamer", "harbour", "water", "archival", "history"], "era": "1910s"}),
        (13, "map_parchment", big, {"title": "Antique map of a coastline", "tags": ["map", "coastline", "isthmus", "route", "history", "archival", "geography"], "era": "1800s"}),
        (14, "map_survey", hd, {"title": "Survey chart with grid", "tags": ["map", "survey", "chart", "terrain", "route", "history"], "era": "1880s"}),
        (15, "night_city", hd, {"title": "City lights at night", "tags": ["city", "night", "skyline", "buildings", "urban", "port city"], "era": "modern"}),
        (16, "cranes_sepia", big, {"title": "Old photograph of construction cranes", "tags": ["construction", "cranes", "workers", "archival", "history", "photograph", "machines"], "era": "1910s"}),
    ]


def render_still(n: int, kind: str, size: tuple[int, int]) -> Image.Image:
    rng = np.random.default_rng(1000 + n)
    w, h = size
    builders = {
        "landscape_dawn": lambda: still_landscape(rng, w, h, (70, 90, 150), (245, 190, 140), (60, 50, 70)),
        "landscape_dusk": lambda: still_landscape(rng, w, h, (40, 30, 70), (230, 120, 90), (30, 40, 35)),
        "landscape_sepia": lambda: age_photo(still_landscape(rng, w, h, (150, 150, 150), (230, 230, 230), (60, 60, 60)), rng),
        "landscape_blue": lambda: still_landscape(rng, w, h, (120, 150, 190), (210, 225, 235), (40, 70, 90)),
        "clouds_day": lambda: still_clouds(rng, w, h, (40, 110, 200), (170, 205, 235)),
        "clouds_storm": lambda: still_clouds(rng, w, h, (40, 45, 55), (120, 125, 130)),
        "lock": lambda: still_industrial(rng, w, h, "lock", (110, 140, 180), (220, 210, 190)),
        "truss": lambda: still_industrial(rng, w, h, "truss", (200, 150, 110), (240, 210, 170)),
        "cranes": lambda: still_industrial(rng, w, h, "cranes", (90, 110, 150), (230, 190, 150)),
        "stacks": lambda: still_industrial(rng, w, h, "stacks", (150, 140, 130), (210, 200, 185)),
        "ship_sepia": lambda: age_photo(still_ship(rng, w, h), rng),
        "ship_mono": lambda: age_photo(still_ship(rng, w, h), rng).convert("L").convert("RGB"),
        "map_parchment": lambda: still_map(rng, w, h),
        "map_survey": lambda: age_photo(still_map(rng, w, h), rng),
        "night_city": lambda: still_night_city(rng, w, h),
        "cranes_sepia": lambda: age_photo(still_industrial(rng, w, h, "cranes", (150, 150, 150), (220, 220, 220)), rng),
    }
    return builders[kind]()


def make_stills() -> None:
    for n, kind, size, meta in still_specs():
        img = label_image(render_still(n, kind, size).convert("RGB"), f"SAMPLE IMAGE {n:02d}")
        out = OUT / f"sample_image_{n:02d}.jpg"
        img.save(out, quality=88, optimize=True)
        desc = f"Synthetic placeholder still: {meta['title'].lower()}."
        write_sidecar(out, meta["title"], meta["tags"], meta["era"], desc, width=size[0], height=size[1])
        print(f"{out.name} {out.stat().st_size // 1024} KB", flush=True)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only-stills", action="store_true")
    ap.add_argument("--only-video", action="store_true")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.only_video:
        make_stills()
    if not args.only_stills:
        make_videos()
    total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    print(f"total {total / 1e6:.1f} MB in {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
