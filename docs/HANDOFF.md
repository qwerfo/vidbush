# Vidbush handoff

Vidbush turns a brief, a script, or a finished voiceover into a publish-ready
long-form documentary or Top 10 video, plus captions, chapters, a description
with credits, three thumbnails, a QA report and a cost report. This file is
self-contained: you do not need any earlier conversation to use it.

## 1. What works, what is mock, what is stubbed

### Verified working in this build (mock mode, no API keys)

| Check | Result |
|---|---|
| A1 `python -m vidbush demo` (Panama Canal brief, 4 min documentary) | Passes every check: about 4.3 min, 40+ shots, 5 shot types, animated maps, bar chart, title cards, number cards, lower thirds, callouts, burned captions, ducked music, chapters, description with credits, 3 thumbnails, QA and cost reports. -14.0 LUFS, about -1.1 dBTP, 1920x1080, 30 fps, H.264 High, AAC 192k. |
| A2 script mode (`fixtures/demo_script.md`, Hoover Dam) | Passes the same checks. |
| A3 voiceover mode (`fixtures/demo_voiceover.mp3`, Erie Canal) | Passes the same checks. The script next to the mp3 is used for alignment. |
| A4 `python -m vidbush serve` | UI and API on port 8080. Create a project, watch stages and log, play the video, open a shot, swap it: only that segment re-renders. Checked in headless Chromium with no console errors. |
| A5 `revise demo "make the first 30 seconds faster and replace shot 7 with an AI image"` | Maps to `retime_section [0, 30] x0.7` then `change_type shot_007 -> ai_image`, applies both, re-renders only changed segments. |
| A6 `learn-style fixtures/reference_video.mp4 test_profile` | Finds 12 shots, the 2.39 letterbox and the caption band; writes `config/style_profiles/test_profile.yaml`. |
| A7 pytest | 620+ unit tests plus the slow demo integration test (`tests/test_integration_demo.py`), which runs the demo twice and asserts the second assembly renders 0 segments. |
| A8 crash and resume | `VIDBUSH_FAIL_AT=judge` stops the run at stage 8; `resume demo` skips intake through sourcing and finishes. |
| A9 Docker | Image builds (ubuntu 24.04 base), `docker compose up` serves the UI and API on 8080, and a project rendered inside the container. |
| A10 em dash | Zero in the repo; `tests/test_core_no_em_dash.py` guards it. |

Project status values are `done`, `needs_review` (finished, but some role you
configured as real fell back to mock), and `failed`.

### What is mock in the demo (and why)

The build sandbox blocked every outside API (Wikipedia, YouTube, Commons,
Archive, LoC, NASA, OpenAI-style APIs, ElevenLabs, Hugging Face). So in the
demo:

- Research, outline, script, titles, description and the director's visual
  plan come from canned fixtures in `fixtures/llm/the_panama_canal/` (written
  to be accurate, with real Wikipedia sources).
- Footage and stills come from a bundled synthetic library
  (`fixtures/sample_assets/`), each visibly labelled SAMPLE FOOTAGE or SAMPLE
  IMAGE.
- The judge scores from image statistics, AI images are labelled gradients,
  and narration is espeak-ng (robotic but intelligible).
- Every mock call is marked in `cost_report.json` and listed in `qa_report.md`.

Motion graphics, captions, music, mixing, loudness, assembly, cache, editor,
chat revisions, thumbnails composition, chapters and the UI are all real code
paths, not mocks.

### Built but not verified against live services

All real adapters are written and unit tested against canned responses, but
were never run live because of the sandbox:

- OpenAI-compatible chat, vision and images (works with OpenRouter and other relays by base URL), fal.ai and Runware images, fal image-to-video.
- ElevenLabs TTS with timestamps, model listing, default premade voice.
- Tavily, Serper, Brave search; Wikipedia.
- YouTube (yt-dlp section downloads plus scene detection), Wikimedia Commons, Internet Archive, Library of Congress, NASA, Pexels, Pixabay, with per-host rate limits and on-disk search caching.
- faster-whisper transcription (model download was blocked).

Run `python -m vidbush doctor --probe` once your keys are in: it sends one tiny
request per configured role and tells you exactly which seat works.

Verified live after the build (internet granted): Archive, Library of Congress
and NASA search and download, YouTube search, Whisper transcription (a
voiceover with no script produced a full video). YouTube downloads need
cookies from servers and VPS IPs: set `YOUTUBE_COOKIES` or
`YOUTUBE_COOKIES_FROM_BROWSER` in `.env`. Set `VIDBUSH_CONTACT` to your email
so Wikipedia and Wikimedia do not rate limit you.

### Stubbed or partial

- DepthFlow: wired, off by default, untested (the CLI was not installable here). Ken Burns is used instead.
- ai_video seat: wired for fal-style image-to-video, off by default (0 percent in the mix), untested live.
- Remotion renderer (`renderer/`): renders the same motion spec and was tested on a title card, but the built-in Python renderer is the default and has the richer look.
- Optional SFX: a sparse synthesized whoosh at some motion graphic starts, off by default (enable with `sfx.json` in the music folder, see section 4).

## 2. First run on Windows 11 (PowerShell)

You received the code as `vidbush-source.zip` (and `vidbush.bundle`, the full
git history). Do every step in a normal PowerShell window (Start menu, type
PowerShell, press Enter). Paste one block at a time and wait for it to finish.

**Step 1. Install the tools.** Paste this whole block:

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Gyan.FFmpeg
winget install -e --id Git.Git
winget install -e --id DenoLand.Deno
winget install -e --id OpenJS.NodeJS.LTS
```

Node is only needed if you want to rebuild the UI or try the Remotion
renderer. Deno is needed for YouTube downloads. Optional, for the free robot
narration in mock mode: install espeak-ng from
https://github.com/espeak-ng/espeak-ng/releases (the .msi file).

**Step 2. Close PowerShell and open a new window** so the new tools are on
your PATH. Check them:

```powershell
python --version
ffmpeg -version
```

**Step 3. Unpack the code to `C:\vidbush`.** Right-click
`vidbush-source.zip`, Extract All, choose `C:\`, so that the file
`C:\vidbush\pyproject.toml` exists. (Or with git:
`git clone C:\Users\<you>\Downloads\vidbush.bundle C:\vidbush`.)

**Step 4. Create the Python environment and install.** Paste as one block:

```powershell
cd C:\vidbush
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned -Force
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[whisper,dev]"
```

Every time you open a new PowerShell window later, run these two lines first:

```powershell
cd C:\vidbush
.\.venv\Scripts\Activate.ps1
```

**Step 5. Create your `.env`.**

```powershell
copy .env.example .env
notepad .env
```

For a mock test, leave it as is. For real videos fill at least:
`LLM_PRIMARY_BASE_URL`, `LLM_PRIMARY_API_KEY`, `LLM_PRIMARY_MODEL` (one
OpenRouter key covers writer, director, editor and vision if you also set
`VISION_BASE_URL`, `VISION_API_KEY`, `VISION_MODEL`), `ELEVENLABS_API_KEY`,
`IMAGE_BASE_URL`, `IMAGE_API_KEY`, `IMAGE_MODEL`, and one of
`SEARCH_API_KEY` (Tavily), `PEXELS_API_KEY`, `PIXABAY_API_KEY`. Save and close.

**Step 6. Check the setup, then run the demo** (takes 5 to 15 minutes):

```powershell
python -m vidbush doctor
python -m vidbush demo
```

The video is at `C:\vidbush\output\demo\final.mp4`. Open
`C:\vidbush\output\demo\qa_report.md` to see what was mock.

**Step 7. Start the editor UI.**

```powershell
python -m vidbush serve
```

Open http://localhost:8080 in your browser. Leave the PowerShell window open
while you use it (Ctrl+C stops the server).

**Step 8. Your first real project.** Fill the keys in `.env`, then:

```powershell
python -m vidbush doctor --probe
```

Every role you configured should say ok. Then create a brief file, for
example `C:\vidbush\briefs\my_first.md`:

```markdown
---
topic: The Johnstown Flood of 1889
target_minutes: 12
format: documentary
audience: older US viewers who love American history
angle: the dam the rich built and the town below it
---
```

and run:

```powershell
python -m vidbush run briefs\my_first.md --brand default
```

Or do the same from the UI (Projects, New project, Brief tab). A pasted
script works the same way (`python -m vidbush run my_script.md`), and so
does a finished voiceover (`python -m vidbush run narration.mp3`; put
`narration.md` with the script next to it for perfect alignment).

## 3. First run with Docker on a Linux VPS

```bash
git clone /path/to/vidbush.bundle vidbush
```

```bash
cd vidbush && cp .env.example .env && nano .env
```

```bash
mkdir -p output data music && docker compose up -d --build
```

```bash
docker compose logs -f vidbush
```

Open `http://<vps-ip>:8080` (put it behind your reverse proxy or an SSH
tunnel: there is no login, it is a single-operator tool). Output, data, your
music and your brands and style profiles are bind mounted, so they survive
rebuilds. Run CLI commands inside the container, for example:

```bash
docker compose exec vidbush python -m vidbush doctor --probe
```

```bash
docker compose exec vidbush python -m vidbush demo
```

The image includes FFmpeg, espeak-ng, deno (for YouTube) and faster-whisper.
The whisper model (about 150 MB) downloads on first use.

## 4. Where things go

- **Music:** any folder of mp3, ogg, wav or m4a files. Point a brand at it
  (`music_folder` in the brand yaml, or `MUSIC_FOLDER` in `.env`). In Docker,
  use `./music` on the host, which is `/app/music` inside. Add a `tracks.json`
  for mood matching: `{"track.mp3": {"mood": "tense", "energy": 0.7, "loop": true}}`.
  Moods used: calm, tense, uplifting, mysterious (hook and climax get uplifting or tense, setup and resolution calm, turns tense). Without it
  tracks are chosen at random (logged). Optional `sfx.json` in the same folder:
  `{"enabled": true, "max_per_minute": 3, "gain_db": -24}`.
- **Brands:** `config/brands/<id>.yaml` (or the Brands page). Fields: voice
  (ElevenLabs voice_id, model `auto`), style_profile_id, music_folder,
  intro_path and outro_path (video files), blocklist (domains), niche_notes
  (read by the writer and director, put your channel voice and rules here),
  language. Use with `--brand <id>`.
- **Style profiles:** `config/style_profiles/<id>.yaml`. Copy
  `documentary_default.yaml` and change the mix, pacing, captions, letterbox,
  music ducking and Ken Burns intensity. `learn-style <url or file> <name>`
  writes one from a reference video.
- **Your own footage library:** set `sources.local_library` in
  `config/vidbush.yaml` to a folder of clips and images. Optional sidecar
  `<file>.json` with `title`, `tags`, `licence`, `author` improves matching.
- **Blocked domains and YouTube:** `sources.blocklist` and
  `sources.allow_youtube` in `config/vidbush.yaml`, or `YOUTUBE_ENABLED` in `.env`.
- **Outputs:** `output/<project_id>/`. Logs: `data/logs/<project_id>.jsonl`.

## 5. Cost expectations per finished minute (real providers)

The demo made these calls for 4.3 minutes of video: 1 research, 1 outline,
about 8 writer calls, director calls per scene, 237 judge calls (each with up
to 3 low-detail keyframes), 90 QA vision calls, 35 TTS chunks (3,531
characters, about 825 characters per minute), and image calls only for
fill-in shots. The demo itself cost $0 (all mock). Estimated real cost per
finished minute:

| Item | Volume per minute | Typical price | Per minute |
|---|---|---|---|
| ElevenLabs narration | about 825 characters | $0.15 to $0.30 per 1k chars | $0.12 to $0.25 |
| Judge (vision) | about 55 calls, 3 small images each | cheap vision model to premium | $0.02 to $0.30 |
| QA pass (vision) | about 21 calls, 1 image each | same | $0.01 to $0.05 |
| Research, script, director, editor (text) | about 15k to 30k tokens | mid to strong model | $0.03 to $0.15 |
| AI images | 0 to 1 fill-in images | $0.02 to $0.04 each, ceiling $0.05 | $0.00 to $0.04 |
| Web search, stock APIs | a few queries | Tavily or free tiers | under $0.01 |
| **Total** | | | **about $0.20 to $0.80** |

Narration and the judge dominate. A cheap vision model for `judge` and
`vision` keeps you near the low end. Every real run writes exact numbers
to `cost_report.json` (`per_minute_usd`); use those after your first video.

## 6. Seats: what to map each role to

Roles live in `config/seats.yaml`. Each role lists providers in order; the
first configured one is used and only a failed request moves to the next.
If every real provider for a role fails, the stage fails (resume later)
unless you set `allow_mock_fallback: true`. Model names below are guidance;
pick whatever is best and cheapest this month.

| Role | Used for | Map it to | Why |
|---|---|---|---|
| researcher | fact cards, anchors, fact check | a strong reasoning model with long context | accuracy matters most here; few calls |
| writer | outline and script, titles, description | your best writing model | quality of narration is the product |
| director | visual ideas per scene, motion graphic specs | a strong model that follows JSON schemas well | one call per scene, structured output |
| editor | chat revisions, beat rewrites | a cheap fast model | short prompts, maps text to fixed ops |
| judge | scoring candidates (many calls) | a cheap vision model | volume; "good enough" bar |
| vision | post-render QA, thumbnails | same as judge | volume |
| cheap_classifier | learn-style shot classification | the cheapest vision model | one call per shot |
| image | AI fill-in shots and thumbnail backgrounds | one consistent photographic model under $0.05 | never mixes styles within a video unless the primary is down |
| video | optional image-to-video | fal-style model | off by default |
| tts | narration | ElevenLabs, model `auto` (multilingual v2) | word timestamps for captions |
| search | research web search | Tavily, Serper or Brave | any one is enough |

Providers marked down after transport failures are retried every 10 minutes.
429 rate limits back off using Retry-After and never mark a provider down.

## 7. Known gaps and the next three improvements

Known gaps:

- No live provider has been exercised (sandbox). Expect small request-shape
  fixes on first contact; the per-model quirk learner handles the common ones.
- The heuristic fallbacks (used only when no LLM answers) give thin scripts
  and generic shot intents. With a real LLM they are not used.
- The bundled music is generated by code (CC0) and only checked by metrics,
  not by ear. Use your own library for real videos.
- The judge cannot detect burned-in captions or logos without a real vision
  model.
- No login on the UI; keep port 8080 private.

Next three improvements, in order:

1. **First live run and calibration:** set keys, `doctor --probe`, run one
   real 10 minute project, then tune the judge threshold, director prompt
   and pacing from that QA report. This is where most of the quality jump
   comes from.
2. **Footage quality filters:** add a burned-in text and logo detector
   (OCR on keyframes) and a face-to-camera detector before the vision judge,
   to cut judge cost and rejects.
3. **Batch and schedule:** a queue page that takes a list of briefs per brand
   and renders overnight, with a per-run cost cap.

See also: `docs/ARCHITECTURE.md` (stage graph, contracts, cache rule),
`docs/DECISIONS.md`, `docs/PROGRESS.md`, `docs/BLOCKERS.md`, `docs/SPEC.md`.
