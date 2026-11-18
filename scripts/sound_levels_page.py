"""A one-page level ladder for the layers Adam has not heard: music under speech, ambience, graphic effects.

Each row is one short excerpt of a planned mix played four times, the row's layer 5 dB apart and everything
else at its current level; each clip is brought to the same loudness so only the balance differs. A free
text comment per row and an Export button that downloads the picks as a JSON file.

  python scripts/sound_levels_page.py [work_dir] [out_dir]
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from vidbush.audio.loudness import SR, normalize, read_audio, write_audio  # noqa: E402
from vidbush.util import run  # noqa: E402

WORK = Path("E:/coding projects/Vidbush/sound/work_new")
OUT = Path("E:/coding projects/Vidbush/sound")
CLIP_S = 12.0


def rows(plan: dict) -> list[dict]:
    lv = plan["levels_db_rel_narration"]
    fx = next((e for e in plan["effects"] if e["kind"] == "line_draw"), plan["effects"][0] if plan["effects"] else None)
    bed = max((b for b in plan["ambience"] if b["place"] != "room_tone"), key=lambda b: b["end"] - b["start"])
    music_t = next(m for m in plan["music"] if m["end"] - m["start"] > 30)["start"] + 8.0
    return [
        {"id": "music", "title": "Music under the narration", "stem": "music", "set": lv["music_under_speech_db"],
         "steps": [lv["music_under_speech_db"] + d for d in (5, 0, -5, -10)], "start": music_t,
         "ask": "How loud should the music sit while the narrator speaks?"},
        {"id": "ambience", "title": f"Ambience under the narration ({bed['place'].replace('_', ' ')})",
         "stem": "ambience", "set": lv["ambience_db"], "steps": [lv["ambience_db"] + d for d in (10, 5, 0, -5)],
         "start": max(bed["start"] + 1.0, bed["start"] + (bed["end"] - bed["start"] - CLIP_S) / 2),
         "ask": "Barely noticeable, never inaudible: which one?"},
        {"id": "effects", "title": f"Graphic effects ({fx['kind'].replace('_', ' ')} at {fx['t']:.1f} s)",
         "stem": "effects", "set": lv["effects_db"], "steps": [lv["effects_db"] + d for d in (5, 0, -5, -10)],
         "start": max(0.0, fx["t"] - 4.0), "ask": "Felt more than heard: which one?"},
    ]


def build(work: Path = WORK, out: Path = OUT) -> Path:
    plan = json.loads((work / "sound_plan.json").read_text(encoding="utf-8"))
    stems = {k: read_audio(work / "stems" / f"{k}.wav", SR, 2) for k in ("narration", "music", "ambience", "effects")}
    clips = out / "levels"
    clips.mkdir(parents=True, exist_ok=True)
    made = []
    for row in rows(plan):
        a, b = int(row["start"] * SR), int((row["start"] + CLIP_S) * SR)
        for i, db in enumerate(row["steps"]):
            parts = {k: v[a:b] * (10 ** ((db - row["set"]) / 20) if k == row["stem"] else 1.0) for k, v in stems.items()}
            mix = sum(parts.values())
            k = int(0.3 * SR)
            ramp = np.linspace(0, 1, k, dtype=np.float32)[:, None]
            mix[:k] *= ramp
            mix[-k:] *= ramp[::-1]
            raw, wav, mp3 = clips / "_raw.wav", clips / "_n.wav", clips / f"{row['id']}_{i + 1}.mp3"
            write_audio(raw, mix, codec="pcm_f32le")
            normalize(raw, wav, -14.0, -1.5, SR, 2)
            run(["ffmpeg", "-v", "error", "-y", "-i", str(wav), "-c:a", "libmp3lame", "-b:a", "192k", str(mp3)])
            raw.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)
        made.append(row)
    page = out / "levels.html"
    page.write_text(render(made), encoding="utf-8")
    return page


def render(rows_: list[dict]) -> str:
    sections = []
    for r in rows_:
        cells = []
        for i, db in enumerate(r["steps"]):
            now = " (now)" if abs(db - r["set"]) < 1e-6 else ""
            cid = f"{r['id']}_{i + 1}"
            cells.append(f"""<div class="clip"><button class="play" data-src="levels/{cid}.mp3">Play</button>
<label><input type="radio" name="{r['id']}" value="{db:g}"> {chr(65 + i)}: {db:g} dB{now}</label></div>""")
        sections.append(f"""<section data-row="{r['id']}" data-set="{r['set']:g}">
<h2>{html.escape(r['title'])}</h2><p>{html.escape(r['ask'])} Levels are dB under the narration, 5 dB apart.
Each clip starts at {r['start']:.1f} s of Alaska run 3.</p>
<div class="row">{''.join(cells)}</div>
<textarea placeholder="Anything about this row (no length limit)"></textarea></section>""")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Vidbush sound levels</title>
<style>
body{{font:17px/1.5 system-ui,sans-serif;max-width:980px;margin:30px auto;padding:0 16px;background:#faf8f4;color:#222}}
section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:14px 18px;margin:18px 0}}
.row{{display:flex;gap:14px;flex-wrap:wrap}} .clip{{border:1px solid #ccc;border-radius:6px;padding:10px;min-width:190px}}
button{{font-size:17px;padding:6px 16px;margin-right:8px;cursor:pointer}} button.playing{{background:#f4d58d}}
textarea{{width:100%;min-height:90px;margin-top:12px;font:16px system-ui}} #export{{font-size:19px;padding:10px 26px}}
@media (prefers-color-scheme: dark){{body{{background:#1d1d1d;color:#eee}} section{{background:#262626;border-color:#444}}
textarea{{background:#1d1d1d;color:#eee}}}}
</style></head><body>
<h1>Sound levels: pick one in each row</h1>
<p>Headphones, one comfortable volume for the whole page. Play the four clips in a row, pick the one that sounds
right, add any comment, then press Export at the bottom. Your picks are also kept in this browser if you close
the page.</p>
{''.join(sections)}
<p><button id="export">Export</button> <span id="done"></span></p>
<audio id="player"></audio>
<script>
const KEY = 'vidbush_levels_v1';
const player = document.getElementById('player');
let current = null;
document.querySelectorAll('button.play').forEach(b => b.onclick = () => {{
  if (current === b && !player.paused) {{ player.pause(); b.classList.remove('playing'); b.textContent = 'Play'; return; }}
  document.querySelectorAll('button.play').forEach(x => {{ x.classList.remove('playing'); x.textContent = 'Play'; }});
  player.src = b.dataset.src; player.play(); current = b; b.classList.add('playing'); b.textContent = 'Stop';
}});
player.onended = () => {{ if (current) {{ current.classList.remove('playing'); current.textContent = 'Play'; }} }};
function state() {{
  return [...document.querySelectorAll('section[data-row]')].map(s => {{
    const r = s.querySelector('input[type=radio]:checked');
    return {{row: s.dataset.row, current_db: Number(s.dataset.set), choice_db: r ? Number(r.value) : null,
             comment: s.querySelector('textarea').value}};
  }});
}}
function save() {{ localStorage.setItem(KEY, JSON.stringify(state())); }}
document.addEventListener('input', save);
try {{
  (JSON.parse(localStorage.getItem(KEY)) || []).forEach(x => {{
    const s = document.querySelector(`section[data-row="${{x.row}}"]`); if (!s) return;
    if (x.choice_db !== null) {{ const r = s.querySelector(`input[value="${{x.choice_db}}"]`); if (r) r.checked = true; }}
    s.querySelector('textarea').value = x.comment || '';
  }});
}} catch (e) {{}}
document.getElementById('export').onclick = () => {{
  const name = 'vidbush_levels_choice.json';
  const data = {{page: 'Vidbush sound levels', exported: new Date().toISOString(), rows: state()}};
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}}));
  a.download = name; document.body.appendChild(a); a.click(); a.remove();
  document.getElementById('done').textContent = 'Saved as ' + name + ' in your Downloads folder ' +
    '(usually C:\\\\Users\\\\<you>\\\\Downloads\\\\' + name + '). Paste that path back.';
}};
</script></body></html>
"""


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    print(build(*args))
