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
  review.json               operator review: one entry per shot (verdict, comment, hard rejects,
                            timestamps, snapshot of the reviewed pick) plus the project comment
  review.md                 review export for the next tuning session (review-export)
  sources/<shot_id>/...     downloaded assets, each with <file>.json sidecar
  inputs/                   copies of the operator's input files
  work/
    research.json outline.json script.json narration.json director_plan.json
    script_summary.json     director's one paragraph summary of the script (keyed by script hash)
    fallbacks.json          heuristic fallbacks per stage while a real provider was configured
    shotlist.json captions.json mix.wav mix.json
    narration/              per-scene tts audio + narration.wav
    mg/                     rendered motion graphics (<hash>.mp4, overlays <hash>.mov)
    ai/                     generated images
    segments/               one mp4 per shot, named <render_hash>.mp4
    render_stats.json       {"rendered": n, "cached": m, "segments": total}
    profile_override.json   optional per-project StyleProfile delta (edit ops)
```

## Batch queue (files on disk)

```
data/batches/<batch_id>/
  state.json                BatchState: source, brand, cost_cap_usd, paused, status, items
                            (input, mode, project_id, status, note, duration, cost, QA percent)
  batch_report.md           one row per project (rewritten whenever an item changes)
  batch_report.json         the same, machine readable, with a summary
  .lock                     held by the runner (same lock module as projects)
output/<id>/work/cost_cap.json   written when the per-run cost cap stopped the project
```

`vidbush/jobs/batch.py` discovers inputs (folder, non recursive, sorted; or a
list file), runs items one at a time through `api/jobs_handler.run_pipeline`
(the worker's own path: `create_project` then `run_project`), records a
failed item and moves on, and checks the paused flag between projects.
`vidbush/jobs/batch_report.py` writes the report. `vidbush/jobs/costcap.py`
is the per-run cap: `run_project(..., cost_cap=usd)` (default None, no cap)
checks `ctx.tracker.total_usd` after each stage and stops at that stage
boundary with the project `needs_review` and `project.status_note` set; a
plain `resume` finishes it. CLI: `batch`, `batch-status`, `batch-pause`,
`batch-resume` (`vidbush/batch_cli.py`).

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
| shots/director.py | `build_shotlist(project, script, narration, research, profile, brand, seats, rng) -> (ShotList, DirectorPlan)` | ideas per beat (director seat called once per scene or batch of 8 beats, in sequence, shots/seat.py; every call carries the script summary from work/script_summary.json, the previous two chunks' ideas, the visual anchors so far and the next beat), then the pacing engine and assembler (shots/assembler.py) assign slots; at most 2 full screen cards and 16 s in a row, and at most one free full screen card per `pacing.card_spacing_s` (75 s) outside the recap (and a top 10 hook), each held at most `pacing.card_hold_max` (4 s, maps and charts 7 s) and started on the word that speaks it; visual mix measured by shot count from `StyleProfile.shot_mix()` |
| source/registry.py | `source_all(shotlist, ctx) -> ShotList`, `source_shot(shot, ctx, queries=None) -> list[Candidate]` | parallel adapters, cap 8, sidecars |
| judge/director_loop.py | `judge_and_select(shotlist, ctx) -> ShotList` | score, select, re-query, convert to ai_image, never empty |
| visuals/prepare.py | `prepare_visuals(shotlist, ctx) -> ShotList` | derived crops, optional DepthFlow and ai_video |
| visuals/ai_images.py | `generate_ai_candidate(shot, ctx, prompt=None) -> Candidate` | image seat, configured provider order |
| mg/renderer.py | `render_spec(spec, out_path, alpha, settings) -> Path`, `render_all_motion(shotlist, ctx) -> (ShotList, list[Path])` | python or remotion, cached by spec hash |
| mg/routes.py | `route_points(a, b, via) -> list[[lat, lon]] or None`, `infer_via(*texts) -> str`, `path_miles(points) -> float` | rail, road, river: snap within 10 km and Dijkstra over the bundled line layer; sea, air: dashed great circle; line (no mode named): rail, then road (`modes(via)`); trail: none |
| render/captions.py | `group_words(words, style) -> list[CaptionLine]`, `build_captions(narration, profile, out_root, settings) -> list[Path]`, `write_ass_window(lines, profile, t0, t1, path, w, h) -> Path`, `caption_geometry(profile, w, h) -> dict` | one line, 3 to 5 words, fixed band height |
| audio/music.py | `build_mix(narration, script, shotlist, brand, profile, settings, work_dir, log, rng) -> Path` | writes work/mix.wav at -14 LUFS, ducked bed |
| render/assemble.py | `assemble(shotlist, ctx) -> (Path, ShotList)` | per-shot segments with cache, concat, intro/outro, mux |
| render/package.py | `package(shotlist, ctx) -> list[Path]` | chapters, description, thumbnails |
| judge/qa.py | `build_qa(shotlist, ctx) -> list[Path]` | post-render judge pass, qa_report.json and .md |
| editor/ops.py | `apply_ops(settings, project_id, ops) -> EditResult`, `apply_and_render(settings, project_id, ops, echo=True) -> EditResult` | fixed EditOp vocabulary |
| editor/revise.py | `map_message(settings, project_id, message) -> list[EditOp]` | editor seat with rule-based fallback |
| review/store.py | `load(settings, id) -> Review`, `update_shot(settings, id, shot, verdict=, comment=, rejects=)`, `set_project_comment`, `mark_fixed`, `progress` | review.json, reconciled with the shot list on every load |
| review/export.py | `summary(review)`, `markdown(review, title, threshold)`, `write_export(settings, id) -> Path` | review.md |
| review/fixes.py | `plan(settings, id) -> {ops, timing, comments, skipped}` | verdicts to EditOps (change_type re-source with add_must_avoid, add_queries, reject_selected; regenerate_ai for AI shots) |
| review/calibrate.py | `calibrate(review, threshold) -> dict`, `calibration_text(c)` | judge scores against the review verdicts, suggested threshold, missed rejects |
| learn/style_learner.py | `learn_style(source, name, settings) -> Path` | writes config/style_profiles/<name>.yaml |

### Motion graphics

`MotionSpec` is the renderer contract. `kind` picks the composition and
`payload` is parsed by `spec.typed_payload()` into `MapPayload`,
`ChartPayload`, `NumberCardPayload`, `TextPayload`, `BoxPayload` or
`RecapPayload`. Full-frame motion graphic shots carry `shot.motion`; the
motion stage renders it to `work/mg/<hash>.mp4` and adds a Candidate with
`source=motion`, which becomes `shot.selected`. Overlay shots carry
`shot.overlay`; the motion stage renders it with alpha to
`work/mg/<hash>.mov` (qtrle, rgba) and sets `shot.overlay_path`. A map or
chart overlay renders as an inset panel over the picture (`vidbush/mg/inset.py`,
always the Python renderer). A text overlay (callout, text_overlay,
lower_third, highlight_box, arrow) carries `anchor` (the narration words it
belongs to) and `start_at` (seconds into the shot): the director sets
`start_at` 0.2 s after the anchor is spoken (`rules.time_overlays`), the
clip renders for `duration - start_at` and the segment overlays it from
`start_at`. The motion
cache key includes the renderer kind and a renderer version; degraded
fallback renders are stored under a separate `.fallback` name.
Data-driven graphics (maps, charts) always draw `source_note` in a corner.
Palettes: `documentary_warm`, `documentary_cold`, `top10_bold`. Font: Inter
(bundled in `assets/fonts`, OFL). Maps use Natural Earth data bundled in
`assets/geo` and never a tile server. Every model-written map (director motion
card or inset, editor motion_spec) passes `shots/seat.verify_map`: markers are
gazetteer places in the map's own view (route endpoints included), routes are
real geometry from `mg/routes.py`, distances are confirmed by the drawn line,
shaded regions exist in admin 0 or admin 1 (a state sharing a country's name is
chosen by the markers or view and stored as `State, Country`), and the source
note is fixed; a map with nothing true left is rejected and the slot gets footage.

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

#### Visual mix semantics

`StyleProfile.visual_mix` holds the picture mix (footage, footage_overlay,
image, image_overlay, ai_image, ai_video) plus two animation numbers:
`full_screen_cards` (motion_graphic shots) and `animated_overlays` (maps,
charts, callouts, arrows and lower thirds over footage or stills). Animated
overlays are counted on top of the shot they decorate: `shot_mix()` adds
them to footage_overlay and image_overlay in the footage to still ratio of
the picture mix. Documentary default: footage 20, footage_overlay 47.2,
image 7, image_overlay 20.8, motion_graphic 5. The old `motion_graphic` key
is read as `full_screen_cards`. QA reports the shot type mix against
`shot_mix()` and an animation table (full screen cards, shots with an
overlay, map and chart insets).

## Captions geometry

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
returns to the primary. A writer section or director chunk that falls back to
heuristics while its role has a real provider is recorded with
`seats.note_fallback(role, stage, scene_id, reason)`; the stage writes the
notes to `work/fallbacks.json`, QA lists them under `heuristic_fallbacks` and
the project ends `needs_review`. 429 responses back off using Retry-After (up to 6
attempts) and never mark a provider down. `options.max_concurrency` limits
parallel requests per provider. If a role has a configured real provider and
every real provider fails, the seat raises and the stage fails (resume
retries later), unless `allow_mock_fallback` is true. A finished project that
used mock for a role configured as real ends as `needs_review`.

Provider types are dispatched through registries in `vidbush/providers/seats.py`:
`LLM_TYPES` (`openai_compat` in `llm.py`, `anthropic` in `llm_anthropic.py`, the
native Messages API) for text and vision roles, and `TTS_TYPES` (`elevenlabs` in
`tts.py`, `algrow_tts` in `tts_algrow.py`, `elevate_tts` in `tts_elevate.py`, one
line per type) for narration. A
role entry naming an unregistered type is skipped with one warning. A role entry
may give `model_match: {family, version}` (plus `prefix: ""` for non Claude
families) instead of a model; the id is then read
once per provider from its `GET /models` (`providers/model_resolve.py`), and a
`model_env` pin naming another family or version is ignored with one warning.
An HTTP error about the seat itself (401, 402, 403, model_not_in_channel, usage
or credit limits) raises `SeatUnavailable` and marks that provider:model down
until the 10 minute probe. Settings read `.env`, then `.env.local`, then the
process environment (the last wins).
`VIDBUSH_PREFER_PROVIDER` (until `VIDBUSH_PREFER_UNTIL`) moves named providers to
the front of every role. LLM replies stream by default (`options.stream`).

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

Review (`vidbush/api/routes_review.py`): `GET /api/projects/{id}/review`,
`PUT /api/projects/{id}/review/shots/{shot}` (`{verdict, comment, rejects, clear_verdict}`, fields left
out stay), `PUT /api/projects/{id}/review/comment`, `GET .../review/summary`, `POST .../review/export`,
`GET .../review/fixes` (the plan), `POST .../review/fixes` (`{ops}`, one apply_ops job) and
`GET .../review/calibration`. `GET /api/projects/{id}` carries `review: {reviewed, total}` once
review.json exists. UI: `ui/src/pages/Review.jsx` at `#/review/<id>`; `#/p/<id>/<shot>` opens a shot's
panel with the duration box focused. CLI: `vidbush/review_cli.py`.

Batches (`vidbush/api/routes_batches.py`): `GET /api/batches`,
`GET /api/batches/{id}` (report rows plus state), `POST /api/batches`
(`{source, brand, profile, format, minutes, cost_cap_usd, id_prefix,
batch_id, start}`, source is a folder or list file on the server),
`POST /api/batches/{id}/pause`, `POST /api/batches/{id}/resume`
(`{retry_failed, cost_cap_usd}`). A batch runs as one worker job of kind
`batch` whose `project_id` is `batch:<id>`, so it occupies one worker slot
until the queue is empty or paused, and a server restart requeues it. The UI
Queue page (`ui/src/pages/Queue.jsx`, `#/queue/<id>`) polls these routes.
