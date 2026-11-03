"""Build fixtures/reference_video.mp4: a synthetic style reference for learn-style tests.

About 45 s at 1280x720, 30 fps, no audio. Twelve hard cuts of visually distinct
lavfi sources (2 to 8 s each), 2.39:1 letterbox bars (92 px at 720p) and a
burned single line caption centred in the bottom bar on most shots (Inter
ExtraBold, white with a black border, 34 px at 720p which is 51 px at 1080p).

Run: python fixtures/make_reference_video.py [out_path]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FONT = ROOT / "assets" / "fonts" / "Inter-ExtraBold.otf"
W, H, FPS = 1280, 720, 30
BAR = round((H - W / 2.39) / 2)  # 92 px
CAPTION_SIZE = 34

# (lavfi source, seconds, caption or "" for no caption)
SHOTS: list[tuple[str, float, str]] = [
    ("testsrc2=size=1280x720:rate=30", 3.2, "The canal opened in 1914"),
    ("mandelbrot=size=1280x720:rate=30", 5.1, "Nobody expected it to work"),
    ("smptehdbars=size=1280x720:rate=30", 2.4, "Ten years of digging"),
    ("gradients=size=1280x720:rate=30:c0=#1a2a44:c1=#e8c070:c2=#6b8f3a:n=3:speed=0.02:type=spiral", 6.8,
     "A jungle full of danger"),
    ("gradients=size=1280x720:rate=30:c0=#0b3d91:c1=#f2a541:speed=0.03", 4.2, ""),
    ("pal75bars=size=1280x720:rate=30,hue=H=t/2", 2.8, "Every ship paid a toll"),
    ("zoneplate=size=640x360:rate=30:kt2=4:kx2=60:ky2=60", 5.4, "The locks lift ships high"),
    ("rgbtestsrc=size=1280x720:rate=30", 3.6, "Steel gates weigh tons"),
    ("color=c=#7a1f1f:size=1280x720:rate=30,drawgrid=w=160:h=120:t=6:c=#e8b4a0", 2.2, "Fever took thousands"),
    ("sierpinski=size=1280x720:rate=30:type=triangle", 4.0, "Then came a new plan"),
    ("gradients=size=1280x720:rate=30:c0=#0f5132:c1=#d1e7dd:speed=0.05:type=radial", 3.0, ""),
    ("testsrc=size=1280x720:rate=30", 2.3, "And the world changed"),
]


def _filters(caption: str) -> str:
    """Letterbox bars and an optional caption centred in the bottom bar."""
    chain = [f"scale={W}:{H}", "format=yuv420p",
             f"drawbox=x=0:y=0:w={W}:h={BAR}:color=black:t=fill",
             f"drawbox=x=0:y={H - BAR}:w={W}:h={BAR}:color=black:t=fill"]
    if caption:
        text = caption.replace(":", "\\:").replace("'", "")
        chain.append(f"drawtext=fontfile='{FONT}':text='{text}':fontsize={CAPTION_SIZE}:fontcolor=white:"
                     f"borderw=3:bordercolor=black:x=(w-text_w)/2:y={H - BAR}+({BAR}-text_h)/2")
    return ",".join(chain)


def _segment(src: str, seconds: float, caption: str, out: Path) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", src, "-t", f"{seconds:.3f}",
           "-vf", _filters(caption), "-r", str(FPS), "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
           "-pix_fmt", "yuv420p", "-g", "60", "-an", str(out)]
    subprocess.run(cmd, check=True)


def build(out: Path) -> Path:
    """Render every shot, then concatenate with stream copy."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (src, secs, cap) in enumerate(SHOTS):
            p = Path(tmp) / f"part_{i:02d}.mp4"
            _segment(src, secs, cap, p)
            parts.append(p)
        lst = Path(tmp) / "list.txt"
        lst.write_text("".join(f"file '{p}'\n" for p in parts), encoding="utf-8")
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
                        "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(out)], check=True)
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "fixtures" / "reference_video.mp4"
    print(build(target))
