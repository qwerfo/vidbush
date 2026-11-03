# Vidbush architecture

Vidbush turns a brief, a script, or a voiceover into a finished long-form
video. It is one Python package (`vidbush/`), a motion graphics renderer
(built-in Python renderer, Remotion project in `renderer/` as an optional
alternative), and a small React UI (`ui/`) served by FastAPI on port 8080.

## Stage graph

```
intake -> research* -> outline* -> script -> narration -> shotlist -> sourcing
       -> judge -> visuals -> motion -> mix -> assembly -> packaging
(* brief mode only; skipped for script and voiceover input)
```

Each stage is a function `stage_<name>(ctx) -> list[output paths]` in
`vidbush/jobs/stages.py`. It reads its inputs from `output/<id>/work/`,
calls one module entry point, and writes its output back to disk. The runner
(`vidbush/jobs/pipeline.py`) records every stage in `project.json`. A stage
is skipped on resume when its record is `done` and all its outputs exist, so
`python -m vidbush resume <id>` continues from the first unfinished stage.
`VIDBUSH_FAIL_AT=<stage>` raises at the start of that stage (test hook).
A stage that runs forces every stage that reads its output (`STAGE_INPUTS`
in `vidbush/jobs/pipeline.py`), so a re-run never leaves stale downstream
outputs. Edit ops persist their force set in `work/edit_plan.json` so a
failed edit render is completed by the next resume. `output/<id>/.lock`
(pid based, stale detection) keeps the CLI and the API worker from running
the same project at once. Absolute paths stored in project and work files
are rebased onto the current project root on load, so projects move freely
between machines and between host and Docker.

## Files on disk (per project)

```
output/<id>/
  project.json              Project
  final.mp4                 1920x1080, 30 fps, H.264 high, AAC 192k, -14 LUFS, -1 dBTP
  captions.srt, captions.ass
  chapters.txt, description.md
  thumbnails/thumb_1.png .. thumb_3.png   (1280x720)
  shotlist.json             ShotList (copy of work/shotlist.json after assembly)
  qa_report.json, qa_report.md
  cost_report.json          CostReport
  sources/<shot_id>/...     downloaded assets, each with <file>.json sidecar
  inputs/                   copies of the operator's input files
  work/
    research.json outline.json script.json narration.json director_plan.json
    shotlist.json captions.json mix.wav mix.json
    narration/              per-scene tts audio + narration.wav
    mg/                     rendered motion graphics (<hash>.mp4, overlays <hash>.mov)
    ai/                     generated images
    segments/               one mp4 per shot, named <render_hash>.mp4
    render_stats.json       {"rendered": n, "cached": m, "segments": total}
    profile_override.json   optional per-project StyleProfile delta (edit ops)
```

## Contracts

All data contracts are pydantic models in `vidbush/models.py`. JSON schemas
are exported to `docs/schemas/*.json` by `python -m vidbush.schemas`.

`StageContext` (`vidbush/jobs/context.py`) carries: `settings`, `project`,
`brand`, `profile` (StyleProfile), `seats` (provider facade), `log`
(JSON lines logger: `ctx.log.info(msg, **fields)`), `tracker` (cost),
helpers `ctx.work`, `ctx.root`, `ctx.work_path(key)`, `ctx.out_path(name)`,
`ctx.load(key, Model)`, `ctx.save(key, obj)`, `ctx.rng(salt)`.

### Module entry points (integration contract)

| Module | Function | Notes |
|---|---|---|
| research/researcher.py | `run_research(brief, brand, seats, settings, log) -> Research` | researcher seat, Wikipedia + web search when online, fact check pass in research/factcheck.py drops contradicted cards |
| script/outline.py | `make_outline(brief, research, brand, seats) -> Outline` | documentary or top10, ~150 wpm sizing |
| script/writer.py | `write_script(outline, research, brief, brand, seats) -> Script` | writer seat, beats tagged with fact ids |
| script/parser.py | `parse_script(text, fmt=None) -> Script` | pasted scripts, any length, infers scene purposes and top10 |
| audio/transcribe.py | `transcribe_words(audio, settings, log) -> list[Word] or None`, `transcribe_to_text(audio, settings, log, seats) -> str` | faster-whisper when available |
| audio/narration.py | `narrate(script, brand, seats, settings, work_dir, log) -> Narration`, `narration_from_voiceover(audio, script, settings, work_dir, log) -> Narration`, `renarrate_beat(...)` | per-scene TTS, 40 ms crossfade stitch, alignment check, -16 LUFS |
| shots/director.py | `build_shotlist(project, script, narration, research, profile, brand, seats, rng) -> (ShotList, DirectorPlan)` | ideas per beat (director seat called once per scene or batch of 8 beats, in parallel, shots/seat.py), then the pacing engine and assembler (shots/assembler.py) assign slots; at most 2 full screen cards and 16 s in a row; visual mix measured by shot count |
| source/registry.py | `source_all(shotlist, ctx) -> ShotList`, `source_shot(shot, ctx, queries=None) -> list[Candidate]` | parallel adapters, cap 8, sidecars |
| judge/director_loop.py | `judge_and_select(shotlist, ctx) -> ShotList` | score, select, re-query, convert to ai_image, never empty |
| visuals/prepare.py | `prepare_visuals(shotlist, ctx) -> ShotList` | derived crops, optional DepthFlow and ai_video |
| visuals/ai_images.py | `generate_ai_candidate(shot, ctx, prompt=None) -> Candidate` | image seat, configured provider order |
| mg/renderer.py | `render_spec(spec, out_path, alpha, settings) -> Path`, `render_all_motion(shotlist, ctx) -> (ShotList, list[Path])` | python or remotion, cached by spec hash |
| render/captions.py | `group_words(words, style) -> list[CaptionLine]`, `build_captions(narration, profile, out_root, settings) -> list[Path]`, `write_ass_window(lines, profile, t0, t1, path, w, h) -> Path`, `caption_geometry(profile, w, h) -> dict` | one line, 3 to 5 words, fixed band height |
| audio/music.py | `build_mix(narration, script, shotlist, brand, profile, settings, work_dir, log, rng) -> Path` | writes work/mix.wav at -14 LUFS, ducked bed |
| render/assemble.py | `assemble(shotlist, ctx) -> (Path, ShotList)` | per-shot segments with cache, concat, intro/outro, mux |
| render/package.py | `package(shotlist, ctx) -> list[Path]` | chapters, description, thumbnails |
| judge/qa.py | `build_qa(shotlist, ctx) -> list[Path]` | post-render judge pass, qa_report.json and .md |
| editor/ops.py | `apply_ops(settings, project_id, ops) -> EditResult`, `apply_and_render(settings, project_id, ops, echo=True) -> EditResult` | fixed EditOp vocabulary |
| editor/revise.py | `map_message(settings, project_id, message) -> list[EditOp]` | editor seat with rule-based fallback |
| learn/style_learner.py | `learn_style(source, name, settings) -> Path` | writes config/style_profiles/<name>.yaml |

### Motion graphics

`MotionSpec` is the renderer contract. `kind` picks the composition and
`payload` is parsed by `spec.typed_payload()` into `MapPayload`,
`ChartPayload`, `NumberCardPayload`, `TextPayload`, `BoxPayload` or
`RecapPayload`. Full-frame motion graphic shots carry `shot.motion`; the
motion stage renders it to `work/mg/<hash>.mp4` and adds a Candidate with
`source=motion`, which becomes `shot.selected`. Overlay shots carry
`shot.overlay`; the motion stage renders it with alpha to
`work/mg/<hash>.mov` (qtrle, rgba) and sets `shot.overlay_path`. The motion
cache key includes the renderer kind and a renderer version; degraded
fallback renders are stored under a separate `.fallback` name.
Data-driven graphics (maps, charts) always draw `source_note` in a corner.
Palettes: `documentary_warm`, `documentary_cold`, `top10_bold`. Font: Inter
(bundled in `assets/fonts`, OFL). Maps use Natural Earth data bundled in
`assets/geo` and never a tile server.

**Renderer swap rule:** the JSON MotionSpec is the stable contract. Any
renderer that turns a MotionSpec into a clip of exactly `duration` seconds
at `width x height`, `fps`, with alpha when asked, can replace the built-in
one by setting `render.renderer` in `config/vidbush.yaml`.

### Shot types and slot families

| Shot type | Visual source | Overlay |
|---|---|---|
| footage | video candidate | none |
| footage_overlay | video candidate | MotionSpec with transparent background |
| image | still candidate, Ken Burns | none |
| image_overlay | still candidate, Ken Burns | MotionSpec |
| ai_image | generated still, Ken Burns | optional |
| ai_video | generated clip | optional |
| motion_graphic | rendered MotionSpec | none |

### Captions geometry

With letterbox on (2.39:1 at 1920x1080) the active picture is 1920x803 and
each band is 138 px. Captions sit in the bottom band, vertical center at
`band_top + band_height * band_px` (default 0.5, so y = 1011). With
letterbox off, the same rule applies to a virtual bottom band of the same
height. One line, 3 to 5 words, never two lines. Captions are burned per
segment (window of the full caption list shifted to segment time) so the
segment cache stays valid.

## Cache rule

Each shot renders to `work/segments/<render_hash>.mp4`. `render_hash` is a
stable hash of: the selected candidate (source, path fingerprint, in/out
points, crop), shot type, duration, overlay spec and overlay clip
fingerprint, Ken Burns parameters, the caption lines inside the shot window,
the caption style, the letterbox and colour settings, and the render
settings (size, fps, crf). Media files are fingerprinted by content (size
plus a hash of the first and last megabyte), never by mtime, so copying a
project does not invalidate its cache. A segment whose file exists is reused. A change
to one shot changes only that shot's hash, so only that segment re-renders.
`work/render_stats.json` records how many segments were rendered and how
many came from cache on the last assembly.

## Providers

Code asks for a seat by role (`researcher`, `writer`, `director`, `judge`,
`editor`, `cheap_classifier`, `image`, `video`, `tts`, `vision`, `search`).
`config/seats.yaml` maps each role to an ordered provider list. One client
(`vidbush/providers/base.py`) wraps every call: 3 attempts with backoff on
transport errors, one attempt then fallback on refusal or 4xx, providers
that fail on transport are marked down and probed again every 10 minutes,
and every call (real or mock) is written to `cost_report.json`. Fallback
moves only the failed request; the configured order is kept, so one video
never mixes image styles unless the primary is down, and the 10 minute probe
returns to the primary. 429 responses back off using Retry-After (up to 6
attempts) and never mark a provider down. `options.max_concurrency` limits
parallel requests per provider. If a role has a configured real provider and
every real provider fails, the seat raises and the stage fails (resume
retries later), unless `allow_mock_fallback` is true. A finished project that
used mock for a role configured as real ends as `needs_review`.

When a role has no configured provider (or offline mode is on), the mock provider answers: canned
LLM fixtures from `fixtures/llm/<fixture_key>/<task>.json` (fixture key is
`slug(topic)` for briefs, the file stem for scripts and voiceovers), or
`None`, in which case every caller has a deterministic heuristic fallback.
Mock vision scores from image statistics, mock image draws a labelled
gradient placeholder, mock TTS uses espeak-ng (or tone bursts).

## API and UI

FastAPI (`vidbush/api/app.py`) exposes projects, stages, logs, shot list,
candidates, edit ops, chat revisions, brands, settings and seat status, and
serves the built UI from `ui/dist`. A worker pool (`vidbush/jobs/runner.py`)
pulls jobs from a SQLite table in `data/vidbush.db`, one job per project at a
time. Everything the UI does is also a CLI subcommand.
