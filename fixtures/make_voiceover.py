"""Generate fixtures/demo_voiceover.mp3 from fixtures/demo_voiceover.md with espeak-ng.

Run: python fixtures/make_voiceover.py [--out path]

Markdown headings, list markers and emphasis are stripped. Each paragraph is
spoken separately (voice en-us, about 150 words per minute) and joined with a
short pause, longer after a heading, then encoded as mp3 128k mono.
If the markdown file is missing a built in text about the Erie Canal is used.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
SR = 22050
WPM = 150
PARAGRAPH_PAUSE_S = 0.6
SCENE_PAUSE_S = 1.1

FALLBACK_TEXT = """# The Ditch That Built a Nation

In 1817, the United States was a young country with a problem. Between the farms of the Great Lakes and the \
markets of New York lay the Appalachian Mountains. Moving a ton of flour overland cost more than the flour \
itself. Many people believed the West would drift away and become a separate nation.

Governor DeWitt Clinton had a bold answer. He would dig a canal across the state of New York, three hundred \
and sixty three miles from Albany to Buffalo. Critics called it Clinton's Ditch. Thomas Jefferson called it a \
little short of madness.

# Digging Through the Wilderness

Work began on the Fourth of July, 1817, in the town of Rome. There were no professional engineers in America \
at the time. The men who built the canal learned on the job, and many of them went on to build the railroads \
that came later.

Thousands of laborers, many of them Irish immigrants, cut through forests and swamps with shovels, axes and \
horses. In the Montezuma marshes, fever killed workers by the hundreds. Near Lockport, crews blasted a \
staircase of five double locks through a wall of solid rock.

The canal was only four feet deep and forty feet wide. Mules walked along a towpath beside the water, pulling \
long wooden boats at about four miles an hour. Eighty three locks lifted and lowered those boats more than five \
hundred feet along the way.

# The Wedding of the Waters

In October 1825, the canal was finished, two years ahead of schedule. Clinton boarded a boat called the Seneca \
Chief in Buffalo and rode the whole length of the canal to New York City. Cannons placed along the route fired \
one after another, passing the news from Buffalo to Manhattan in about ninety minutes.

At the mouth of the harbor, Clinton poured a keg of water from Lake Erie into the Atlantic Ocean. People called \
it the Wedding of the Waters. The ditch that critics had mocked was now the most important route in the country.

# A Continent Opens Up

The results were immediate. The cost of shipping goods from Buffalo to New York fell by about ninety percent. \
The journey that once took weeks now took about ten days. Farmers in Ohio and Michigan could suddenly sell their \
crops on the coast, and settlers poured west along the water.

New York City grew into the greatest port in America. Towns like Syracuse, Rochester and Buffalo rose along the \
canal. The tolls paid back the entire cost of construction in less than ten years.

The canal carried more than goods. Ideas, newspapers and new religious movements travelled along it so quickly \
that one stretch of upstate New York became known as the Burned Over District. Reformers, preachers and \
abolitionists moved from town to town by boat, and the Underground Railroad used the same waterway to guide \
people toward freedom in Canada.

# Legacy

Railroads eventually took most of the traffic, and the canal was widened and rebuilt more than once. But the \
Erie Canal had already done its work. It tied the young nation together and proved that Americans could build \
something enormous on their own.

Today, pleasure boats and kayaks drift along the same route. Two hundred years later, the little ditch still \
carries water across New York, a quiet reminder of the moment the country chose to grow together.
"""


def paragraphs(markdown: str) -> list[tuple[str, bool]]:
    """(spoken text, follows_heading) for each paragraph; headings become scene breaks."""
    out: list[tuple[str, bool]] = []
    after_heading = False
    for block in re.split(r"\n\s*\n", markdown):
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        body: list[str] = []
        for ln in lines:
            if ln.startswith("#") or re.match(r"^(---|\*\*\*|<!--)", ln):
                after_heading = True
                continue
            body.append(re.sub(r"^([-*+]|\d+\.)\s+", "", ln))
        text = re.sub(r"[*_`>\[\]]", "", " ".join(body)).strip()
        if text:
            out.append((text, after_heading))
            after_heading = False
    return out


def speak(text: str, path: Path) -> None:
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if not exe:
        raise SystemExit("espeak-ng is required: apt install espeak-ng")
    subprocess.run([exe, "-v", "en-us", "-s", str(WPM), "-w", str(path), text], check=True, capture_output=True)


def read_pcm(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1:
            raise SystemExit(f"unexpected espeak format in {path}")
        return w.readframes(w.getnframes())


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * int(seconds * SR)


def build(text_md: str, out: Path) -> Path:
    """Speak every paragraph, join with pauses, encode mp3 128k mono."""
    pcm = bytearray(silence(0.4))
    with tempfile.TemporaryDirectory() as tmp:
        for i, (para, new_scene) in enumerate(paragraphs(text_md)):
            if i:
                pcm += silence(SCENE_PAUSE_S if new_scene else PARAGRAPH_PAUSE_S)
            wav = Path(tmp) / f"p{i}.wav"
            speak(para, wav)
            pcm += read_pcm(wav)
        pcm += silence(0.8)
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-",
                        "-af", "aresample=44100", "-ar", "44100", "-ac", "1", "-c:a", "libmp3lame", "-b:a", "128k",
                        str(out)], input=bytes(pcm), check=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(HERE / "demo_voiceover.md"))
    ap.add_argument("--out", default=str(HERE / "demo_voiceover.mp3"))
    args = ap.parse_args()
    src = Path(args.src)
    text = src.read_text(encoding="utf-8") if src.exists() else FALLBACK_TEXT
    if not src.exists():
        print(f"{src} not found, using the built in Erie Canal text")
    out = build(text, Path(args.out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
