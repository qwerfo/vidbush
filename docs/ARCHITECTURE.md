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
outputs. A run stopped by a failure or the cost cap sets the forced later
stages back to pending, so the next resume reruns them. Edit ops persist their force set in `work/edit_plan.json` so a
failed edit render is completed by the next resume. `output/<id>/.lock`
(pid based, stale detection) keeps the CLI and the API worker from running
the same project at once. Absolute paths stored in project and work files
are rebased onto the current project root on load, so projects move freely
between machines and between host and Docker.

**Shot list snapshots.** Six stages write the shot list: shotlist, sourcing,
judge, visuals, motion and assembly (`SHOTLIST_STAGES` in
`vidbush/jobs/context.py`). Each one writes its own snapshot
`work/<stage>.shotlist.json`, records its hash in `work/snapshots.json`, and
then copies the same bytes to the working copy `work/shotlist.json`. A stage
in that chain reads the snapshot of the stage before it
(`StageContext.shotlist_source`). So `resume <id> --force sourcing`
re-sources from the director's plan, and `--force judge` re-scores from the
sourcing output, never from what later stages made of it. When the working
copy no longer matches the last recorded hash, an edit op or the operator
changed it, and the edit wins: a stage that was not forced reads the working
copy. A forced stage still starts from its input snapshot, but first keeps the
edited copy as `work/operator_edit_<ts>.shotlist.json` (one warning in the
log). Projects made before snapshots, and stages outside the chain (mix,
packaging), read the working copy. A forced judge on such an old project
clears the old verdicts and generated candidates first.

**Run provenance.** Every `run_project` call appends a `RunRecord` to
`project.runs` (`vidbush/jobs/provenance.py`): git commit, branch, dirty
flag, argv, forced stages, cost cap and settings (seat models per role,
hashes of `vidbush.yaml`, `seats.yaml`, the style profile and the brand,
seed, offline, allow_mock_fallback, reply_cache, judge threshold). Never key
values. Every stage attempt is kept in `StageRecord.history` (`StageAttempt`:
times, seconds, status done, failed or capped, error, commit), so a project
resumed on newer code shows which attempt ran on which commit.

## Files on disk (per project)

```
output/<id>/
  project.json              Project, with runs (one RunRecord per run) and per stage history
                            (one StageAttempt per attempt)
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
  audit/                    tile audit of final.mp4 (python -m vidbush audit <id>)
    audit.jsonl             one row per 2 s tile, appended as each finishes, compacted to the latest row per tile
    audit.md                the six numbers against the benchmark and targets, then the ten worst stretches
    audit_summary.json      the six numbers, tile counts, video, model, cost (read by QA, acceptance, batch report)
    audit_cost_report.json  the audit's own cost report (never the project's cost_report.json)
    tiles/                  frames t<ms>.jpg, plus video.json (identity of the video the frames came from)
  work/
    research.json outline.json script.json narration.json director_plan.json
    script_summary.json     director's one paragraph summary of the script (keyed by script hash)
    fallbacks.json          heuristic fallbacks per stage while a real provider was configured
    shotlist.json captions.json mix.wav mix.json
    sound_plan.json         every music section, change, quiet, bed and effect with time, file, level, reason, plus qa
    sound_qa.md             the Sound section of qa_report.md
    stems/                  narration, music, ambience, effects wav (the planned mix's layers)
    <stage>.shotlist.json   snapshot written by each shot list stage (shotlist, sourcing, judge, visuals,
                            motion, assembly); the next stage reads it
    snapshots.json          {"last": stage, "sha": hash of the working copy it wrote, "stages": {...}}
    operator_edit_<ts>.shotlist.json   an edited working copy kept when a forced stage started from its snapshot
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
                            (input, mode, project_id, status, note, duration, cost, pre-render judge
                            percent, audit noticeable percent)
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
is the per-run cap: `run_project(..., cost_cap=usd)` (default None, no cap).
The cap is checked before every live paid call (LLM, vision, image, TTS,
search, video): `Seats.check_cap` raises `CostCapReached`, a BaseException so
no fallback can swallow it. A reply cache hit is free and is not checked. Cost
is recorded when a call returns, so with N calls in flight the overshoot is at
most N - 1 calls. A cap reached inside a stage leaves that stage pending
(`costcap.stopped_inside`); the check after each stage stops at the boundary.
Either way the project ends `needs_review` with `project.status_note` set and
`work/cost_cap.json` written, and forced later stages go back to pending. A
plain `resume` finishes it, and the replies already paid for come back from
the reply cache. `run` and `resume` take `--cost-cap` too. The batch report
shows the pre-render judge percent (bookkeeping only) and the audit
noticeable percent from `audit/audit_summary.json`. CLI: `batch`,
`batch-status`, `batch-pause`, `batch-resume` (`vidbush/batch_cli.py`).

## Clause-first planner (vidbush/plan, profile planner: clause)

Selected when the style profile sets `planner: clause`; `shots/director.py build_shotlist` hands over to
`plan/planner.py plan_shotlist`. The slot planner (`shots/pacing.py`, `assembler.py`, `rules.py`) stays the default.

1. `clauses.py` (code): narration words to clauses from punctuation and timings; one cut per clause at its first
   word minus 0.1 s. Writes `work/clauses.json` (id, beat, scene, text, start, end, words).
2. `direct.py` (director seat, one call per scene): scene purpose, then per clause subject, treatment (picture,
   hold, or one of the eleven layout kinds in `vocab.py`), presentation for route_map, chart, table, timeline and document_focus (plus a document purpose: evidence, period or subject, and the passages to mark),
   real_thing, medium, anchor, queries per source class, layout elements, grammar labels (sequence, callback,
   hero, quiet). `validate` checks timing, reading time and hold limits only; problems go back to the model.
3. `edit.py` (vision seat, one call per scene, after sourcing): numbered contact sheets (`sheets.py`) of the
   scene's pool; the model picks per shot by comparison, names the in-point frame, answers nothing fits with a
   fallback, labels only real pictures. `check` and `resolve` enforce the AI still rulings (never for a real person,
   event or document; at most 2 in a row; at most 4 s), asset repeats and labels. Not yet wired into the judge
   stage: a live run still picks per shot with the judge; the replay runs the edit on stored candidates.
4. `emit.py`: one Shot per clause or held clause group, in the existing ShotList contract; layouts in
   `Shot.layout` (models.LayoutSpec), full-frame layouts as motion_graphic with a title_card placeholder.
   `work/plan_v2.json` keeps everything the models decided.

Replay on a stored project with no sourcing and no render: `python -m vidbush.plan.replay <project> <out>
[--scenes s01,s08] [--env <dir with .env>]` writes clauses, plan, shot list and `metrics.json` (cuts per minute,
anchor inside window and lead, treatments, AI runs, nothing fits, old against new); replies are cached under
`<out>/cache`. `python -m vidbush.plan.compare <project> <out> <scene> <dir>` draws the old against new review
sheet.

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
| visuals/ai_images.py | `generate_ai_candidate(shot, ctx, prompt=None, problem='') -> Candidate`, `scene_brief(shot, ctx) -> dict`, `shot_window_text(shot, ctx)`, `is_stub(shot, title)`, `one_moment(text)` | prompt from a scene brief of the shot's narration window (agent, action, object, place, era, people, one moment); stub intents never become prompts; composed for the letterbox band; named people shown as context and flagged `real_person_photo_wanted`; names removed by the editor seat (`safe_prompt`); brief and prompt stored on the candidate |
| judge/ai_checks.py | `check_still(shot, cand, ctx) -> dict` | one vision call per AI still, seven fixed ok or not questions (subject, count, setting, one image, no text, no named face, framing) on the still cropped to the letterbox band; a not-ok answer sets `reject_reason` so `director_loop.add_ai` regenerates with the reason as context |
| mg/renderer.py | `render_spec(spec, out_path, alpha, settings) -> Path`, `render_all_motion(shotlist, ctx) -> (ShotList, list[Path])` | python or remotion, cached by spec hash |
| mg/routes.py | `route_points(a, b, via) -> list[[lat, lon]] or None`, `infer_via(*texts) -> str`, `path_miles(points) -> float` | rail, road, river: snap within 10 km and Dijkstra over the bundled line layer; sea, air: dashed great circle; line (no mode named): rail, then road (`modes(via)`); trail: none |
| render/captions.py | `group_words(words, style) -> list[CaptionLine]`, `build_captions(narration, profile, out_root, settings) -> list[Path]`, `write_ass_window(lines, profile, t0, t1, path, w, h) -> Path`, `caption_geometry(profile, w, h) -> dict` | one line, 3 to 5 words, fixed band height |
| audio/music.py | `build_mix(narration, script, shotlist, brand, profile, settings, work_dir, log, rng) -> Path` | writes work/mix.wav at -14 LUFS: the planned mix (audio/mixer.py) by default, the single ducked bed when profile.sound.planned is false |
| audio/sound_plan.py | `scene_infos`, `quiet_spans`, `plan_music`, `music_gain_points`, `quiet_gain_points`, `place_of`, `plan_ambience`, `room_tone_spans`, `plan_effects` | the sound plan as data: music per scene purpose (crossfade on a phrase, changes at quiets), quiet fades, ambience from shot descriptions, word-timed graphic effects |
| audio/mixer.py | `build_sound(narration, script, shotlist, brand, profile, settings, work_dir, log) -> Path` | renders the plan from eased gain curves (no compressor), beds levelled and de-looped, writes mix.wav, mix.json, sound_plan.json, sound_qa.md, stems/ |
| audio/sound_qa.py | `measure_mix(work_dir, plan, speech, quiets, words)`, `qa_markdown(plan)`, `qa_section(work_dir)` | levels against the narration and faults: gap, cut, fast ramp, loud bed, effect on a word, loudness |
| audio/library.py | `ensure(settings, category, key, log, variant) -> Sound or None`, `pick`, `candidates`, `load_catalogue` | labelled ambience beds and effects at settings.sound_library; reviewed first, refused never; generates and labels what is missing |
| audio/elevenlabs_sfx.py | `generate(text, seconds, prompt_influence, settings) -> (mp3 bytes, credits)` | ElevenLabs sound effects (ported from RadioPlay) |
| audio/tracks.py | `analyse_track(path)`, `build_tracks_json(folder, write)`, `needs_refresh(folder)` | measures tempo, phrase, energy, brightness, mood for tracks.json; checked fields survive |
| render/assemble.py | `assemble(shotlist, ctx) -> (Path, ShotList)` | per-shot segments with cache, concat, intro/outro, mux |
| render/package.py | `package(shotlist, ctx) -> list[Path]` | chapters, description, thumbnails |
| judge/qa.py | `build_qa(shotlist, ctx) -> list[Path]`, `judge_rendered(shotlist, frames, ctx, words)`, `local_checks(...)`, `overlay_text_stats(shotlist, words)`, `screen_time(shotlist)` | post-render frame check: frames from each shot's own window (`shot_times`), judged against the narration spoken during the shot with named checks plus code checks (text timing, repeats, filler, same picture); screen time by what is on screen (selected candidate, not planned type); every on-screen text item measured; qa_report.md opens with the tile audit's six numbers (`load_audit`, `audit_lines`, from audit/audit_summary.json, or a line saying how to run it), then a five-line summary; `pct_above_threshold` is the pre-render judge, kept as bookkeeping |
| audit/auditor.py, audit/core.py | `run_audit(video, words, out, vision, duration, shots=...)`, `write_reports(rows, out, ...)`, `core.summarize(rows)`, `core.compare(audit, ref)` | tile audit: one frame every 2 s judged by the `auditor` seat against the words heard; rows keyed on prompt, model, words, source line and video identity |
| audit/calibrate.py, audit/plan_replay.py | `build_sheet(...)`, `agreement(marks, key)`; `plan_report(work, shotlist=None)` | 40 tile calibration page and agreement (trusted at 34 of 40); plan replay measures a stored shot plan without rendering |
| scripts/qa_calibration.py | `review_labels(review)`, `compare(...)` | runs the frame check over a finished render read only and compares it with a frame review (recall, false alarms); scripts/regen_ai_shots.py regenerates chosen AI shots into a scratch copy |
| editor/ops.py | `apply_ops(settings, project_id, ops) -> EditResult`, `apply_and_render(settings, project_id, ops, echo=True) -> EditResult` | fixed EditOp vocabulary |
| editor/revise.py | `map_message(settings, project_id, message) -> list[EditOp]` | editor seat with rule-based fallback |
| review/store.py | `load(settings, id) -> Review`, `update_shot(settings, id, shot, verdict=, comment=, rejects=)`, `set_project_comment`, `mark_fixed`, `progress` | review.json, reconciled with the shot list on every load |
| review/export.py | `summary(review)`, `markdown(review, title, threshold)`, `write_export(settings, id) -> Path` | review.md |
| review/fixes.py | `plan(settings, id) -> {ops, timing, comments, skipped}` | verdicts to EditOps (change_type re-source with add_must_avoid, add_queries, reject_selected, operator_wish; regenerate_ai for AI shots); a comment is split by `comment_parts` into a wish (Shot.operator_wish, shown to the judge, and a query) and a complaint (must_avoid), never written whole into must_avoid |
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
always the Python renderer). Every overlay that belongs to words (callout,
text_overlay, lower_third, number_card, and map and chart insets) carries
`anchor` (the narration words it belongs to) and `start_at` (seconds into the
shot): the director sets `start_at` 0.2 s after its content is spoken
(`ideas.spoken_times`: a text's anchor or title, a number, a map's first
drawn place, a chart's first number or label; `rules.time_overlays`), with
2.5 s (text) or 4 s (insets) visible before the cut, else it moves or is
dropped. The clip renders for `duration - start_at` and the segment overlays
it from `start_at`. highlight_box and arrow are no longer proposed by the
director or the heuristics (they never saw the picture); the renderer still
draws them for old shot lists. Picture ideas carry `VisualIdea.anchor` too:
`_Assembler.pick` fills a picture slot with the idea whose words are spoken
in it (`ideas.idea_anchor_time`, `ANCHOR_W`), and every later re-pick goes
through `_Assembler.choose`. `assembler.map_conflict` is the one spacing and
novelty test for every map and chart on screen, and `rules.screen_guard`
runs last over what is visible (text repeats, map_conflict, lower thirds
only for names in `must_show`, no filler subtitles, one label per name per
minute). A map title is `builders.drawn_title` (built from what is drawn).
In `mg/mapdraw.py` one collision set places the credit, marker dots, route
and marker chips, title, locator (zoom 5 or more) and all labels. The motion
cache key includes the renderer kind and a renderer version; degraded
fallback renders are stored under a separate `.fallback` name.
Data-driven graphics (maps, charts) always draw `source_note` in a corner.
Palettes: `documentary_warm`, `documentary_cold`, `top10_bold`. Font: Inter
(bundled in `assets/fonts`, OFL). Maps use Natural Earth data bundled in
`assets/geo` and never a tile server; small US towns a map names explicitly
resolve from USGS GNIS populated places (`assets/geo/gnis_places.json.gz`,
public domain, `geo.lookup(..., small=True, near=...)`, never free text). Every model-written map (director motion
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

### Designed layouts and panels (V2, `vidbush/mg/layouts`)

The planner writes `Shot.layout` (a `LayoutSpec`): one of eleven kinds (`split_comparison`, `number_over_footage`,
`name_card`, `list_slideshow`, `date_title`, `route_map`, `equals_card`, `chart`, `table`, `timeline`,
`document_focus`), its elements in spoken order (`role`, `text`, `anchor` = the exact narration words, `picture` =
a candidate id or path, `stage` 1 or 2), an optional `title`, a `presentation` for the four panel kinds (route_map,
chart, table, timeline: `full`, `panel`, `full_then_dock`, `panel_then_expand`) and an `id` for a timeline that
returns. Role names per kind are in `models.LAYOUT_ROLES`. The motion stage calls
`layouts.stage.render_shot_layouts` before the old motion jobs; a shot with a layout skips the old path.

| Module | Role |
|---|---|
| `mg/house.py` | The house style (ink ground, paper type, amber signal, teal second voice, Inter, paper frames, drift), overridable per brand (`Brand.house_style`). Type floors in output pixels with the reason. |
| `layouts/timing.py` | `anchor_span` (token match, hyphens split, nearest the shot) and `cue_times`: per element `appear` (0.25 s before its first word, never more than 0.3 s), `word`, `word_end`, clip relative. |
| `layouts/base.py` | `LayoutJob` (spec, cues, pictures, frame profile, house, extra), `LayoutComp` (draws at its own size; `fs()` never returns under the floor; `box` excludes letterbox bars and caption band), `Footage` (a real clip or still at a size), `framed`, `drift`. |
| `layouts/jobs.py` | `make_job`: cues from the narration, pictures from the candidate pool (generated pictures refused), nearby word timings in `extra`. |
| `layouts/pictures.py` | split_comparison, number_over_footage, name_card (and `NameLowerThird`, its fallback), list_slideshow, date_title, equals_card. |
| `layouts/data.py` | chart (bars or line), table, timeline (`extra["history"]` holds a returning timeline's earlier events); content scales to its box; `panel_share` sizes a panel from its content at the floors. |
| `layouts/routemap.py` | route_map on `MapComp` (house chips, amber line): places on their words, the line drawn between the two words over real geometry (`mg/routes.py`), places geocoded and checked like `verify_map`, a registered painted base (art lane) when one holds the view. |
| `layouts/document.py` | document_focus: OCR (prefilter RapidOCR, tiled, cached) finds the quote; whole page, move in, dim, underline as read; falls back to a push with the quote beside it. |
| `layouts/panel.py` | `PanelComp`: the four presentations; content laid out at the large and the panel size, cross-faded during the 0.8 s size change; dock and expand timing; uint8 compositing. |
| `layouts/placement.py` | Reads the picture beneath (faces, text, detail, the subject's heart) at sampled times and picks the calmest clear position of six; previous position penalised. |
| `layouts/stage.py` | Per shot: drop a layout whose words are not spoken in it; place panels (smaller sizes tried, else full); full layouts become the selected `motion` candidate; panels become an alpha overlay starting at `Shot.layout_at` and ending 2.5 s after the last element; name card without a real portrait becomes a lower third (never on a generated picture); old map and chart insets become placed panels (`legacy_panel`). Decisions go to `Shot.layout_notes`. Clips are cached by a hash of everything that changes pixels and `LAYOUT_RENDER_VERSION`. |

Removed on the old path (`mg/renderer.py`, `render/mg_python.py`): bare full-frame title, text and callout cards (the shot
shows its best real picture), centre callouts, label boxes visible before their text.

Stills (`render/stillmotion.py`, `render/cache.still_motion`): a destination move from the picture (hold, reveal from
a detail, travel toward the subject, drift), a planner labelled `Shot.still_sequence` (whole, closer, detail of one
photograph, capped at 1.5x native), and stills narrower than half the frame framed on the house ground
(`segment.framed_image`). Matched look (`render/look.py`): `colour.match_look` sets `Shot.grade`, an ffmpeg filter
applied to real sources in the segment. `RENDER_VERSION` seg-7.

### Art layer (run 3, `vidbush/art`)

Designed visuals that are openly illustrations or animations and are counted as animation, never as
photographs: source values `illustrated_map`, `illustration` and `art_video` (photo-style AI stills stay
`ai_image` and `ai_video`). `visuals/prepare.prepare_visuals` calls `art.plan.apply_art(shotlist, ctx)` after the
judge. It does nothing unless the style profile's `art` block (`models.ArtProfile`, every share off by default)
asks for it and the image seat has an Elevate provider. Every decision is written to `work/art/art_report.json`.

| Module | Role |
|---|---|
| `art/mapref.py` | `reference(payload)`: the exact map view (same projection, center, widest zoom) drawn as land, water, lakes, major rivers and borders, with no text. `land_mask`, `register(painting, payload)`: colour classes learnt from the painting itself, a small scale and shift refit, then the largest wrong patch (disagreements thinner than 1.2 percent of the width removed; fail above 4 percent) and land overlap (0.9). Inland water does not count as coast. |
| `art/paint.py` | `paint_reason(spec, window)`: None when a map establishes a place; a reason (route, arrow, shaded region, distance or border words, no coastline) keeps it plain. `map_prompt`: the director seat writes land, water and touches from the narration and the map's places; names and digits are stripped; fixed rules are added. `paint_map`: three variants (two `art.map_model`, one `art.map_fallback_model`), registration and OCR first, then a vision call on appeal and fit. The winner is warped onto the true view and saved with a sidecar naming that view. |
| `art/hooks.py` | Two single-line hooks in `mg/mapdraw.py`. `painted_base` swaps `MapBase.plain` for the painting when the sidecar vouches for this exact view, and draws true borders and river names back on top. `moving_base` plays a checked clip of the painting as the base (cropped to the push-in, only with no shaded region and a push-in of at most 1.5x). Markers, routes, labels, title and source note stay the renderer's own. |
| `art/illustrate.py` | `choose`: the director picks moments no footage can show (process, comparison, scale, before cameras); it replaces only AI photo-style stills, low scoring picks or empty slots, and never draws maps, charts, documents or signs. `illustrate`: the brand's `illustration_style` line comes first, names are removed, two variants, OCR, then the normal judge against the shot window. |
| `art/clips.py` | `plan_clips`: one clip per `art.clip_every_s`, the opening first, from illustrations, painted maps and (when allowed) AI stills, never from real photographs. `make_clip`: video seat, then the first frame must match the still, a vision call answers fixed yes or no questions (same subject, same layout, no new objects, no text, no morphing), and a map clip's last frame must register. `compose` holds the still, then moves for 3 to 5 s. A failed clip keeps the still. |
| `art/elevate.py` | Elevate image call with `reference_images` (RadioPlay's shape), raw and uncropped. A running task is resumed, never paid twice. `usage`/`hour_credits` for the shared hourly budget. |
| `art/plan.py` | `apply_art`: maps (share of map slots, full screen first, a repeat window per view with reuse outside it), then illustrations, then clips. `art_counts(shotlist)` for reports. |
| `art/prove.py`, `art/sheet.py` | Proving run on a finished project's stored files (read only), then the contact sheet, numbered clips and yield numbers. |

Video seat: provider type `elevate_video` (`providers/video_elevate.py`, registered in `seats.VIDEO_TYPES`, probed by
`doctor --probe`), model `veo_3_1`, 1080p by default.

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
`shot_mix()` by rendered type (a shot whose pick is an AI still counts as
`ai_image` whatever it was planned as), screen time by what is really on
screen (real footage, real still, AI still, AI video, graphic) and an animation table (full screen cards, shots with an
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

### Reply cache

Every LLM and vision reply (`Seats.llm_json`, `llm_text`, `vision_json`) is
stored by `vidbush/providers/reply_cache.py`, so a replay with unchanged
prompts costs nothing and returns the same decisions.

- Location: `reply_cache_folder` (env `VIDBUSH_REPLY_CACHE_FOLDER`), by
  default beside the source cache: `<parent of source cache>/reply_cache`
  (`data/reply_cache` with the default `data/source_cache`). One file per key:
  `<folder>/<role>/<key[:2]>/<key>.json`.
- Key: sha256 of role, model, system prompt, user prompt (text, or the image
  parts with their base64 data) and the options that change the reply (json
  mode, temperature).
- N-th occurrence rule: the n-th identical request in one process gets the
  reply stored as occurrence n. A caller that retries the same prompt gets
  the second stored reply, or a live call when none was stored, so a retry
  never loops on one cached answer. A failed occurrence leaves a gap, and a
  replay asks live for it. `fresh=True` (the judge retry after an unusable
  reply) asks live and overwrites that occurrence.
- Only real, non-empty replies are stored, never mock output. A hit is free
  and skips the cost cap check.
- Switch: on by default. Off with the global flag `--no-reply-cache` (before
  the subcommand), `VIDBUSH_REPLY_CACHE=0`, or `reply_cache: false` in
  `config/vidbush.yaml`. Offline mode never uses it.
- Caveat: the key ignores provider options such as `max_tokens`. Clear the
  cache folder, or run with `--no-reply-cache`, after changing an output cap.
- Images, TTS, search and video are not in this cache (the source cache and
  the narration chunk cache cover those).

## Providers

Code asks for a seat by role (`researcher`, `writer`, `director`, `judge`,
`editor`, `cheap_classifier`, `image`, `video`, `tts`, `vision`, `search`,
`auditor`). The `auditor` role is the tile audit's vision seat (Sonnet 5.5
first); it never runs inside the pipeline. LLM and vision replies go through
the reply cache (Cache rule above), and every live call checks the cost cap
first (Batch queue above).
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
native Messages API) for text and vision roles, and `VIDEO_TYPES` (`elevate_video` in `video_elevate.py`; other video types use `video.py`), and `TTS_TYPES` (`elevenlabs` in
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
