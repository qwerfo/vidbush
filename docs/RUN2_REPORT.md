# Vidbush status report: live run 2 (2026-11-10 to 2026-11-12)

Written for Adam and for whoever plans the next session (a person or another AI). It covers every session since Adam's
instruction of 2026-11-11 (contact, run 1 feedback, seat policy, animation research) and the two merge requests and the
Live Run 2 brief just before it. It is meant to be pasted whole into a planning tool, together with the files listed
in section 0.

Branch: `live-run-2` at commit `7b8ad4d` (plus this report), pushed to GitHub (`qwerfo/vidbush`). `main` is still
at `6287e3e`: none of the run 2 work is on `main` yet.

---

## 0. Files to paste alongside this one

Paste these if the planner needs the full picture. Local paths are on Adam's laptop; GitHub links point at the
`live-run-2` branch.

| File | What it holds | Local path | GitHub |
|---|---|---|---|
| This report | Everything below | `C:\projects\vidbush_tmp\docs\RUN2_REPORT.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/RUN2_REPORT.md |
| DECISIONS.md | One line per decision, every lane, every run (347 lines). Sections "Run 2: ..." are new | `C:\projects\vidbush_tmp\docs\DECISIONS.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/DECISIONS.md |
| HANDOFF.md | Operator manual: install, run, seats, review, costs (815 lines). Section 10 covers seats and TTS | `C:\projects\vidbush_tmp\docs\HANDOFF.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/HANDOFF.md |
| PROVIDER_NOTES.md | Every provider quirk seen live, including the 2026-11-11 probe | `C:\projects\vidbush_tmp\docs\PROVIDER_NOTES.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/PROVIDER_NOTES.md |
| ARCHITECTURE.md | Module map, data flow, models | `C:\projects\vidbush_tmp\docs\ARCHITECTURE.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/ARCHITECTURE.md |
| SPEC.md | Original product spec and acceptance criteria | `C:\projects\vidbush_tmp\docs\SPEC.md` | https://github.com/qwerfo/vidbush/blob/live-run-2/docs/SPEC.md |
| PROGRESS.md, BLOCKERS.md | Older build log and blockers (not updated this run) | `C:\projects\vidbush_tmp\docs\` | https://github.com/qwerfo/vidbush/tree/live-run-2/docs |
| Run 2 QA report | Machine QA of the run 2 render | `E:\coding projects\Vidbush\output\alaska_run2\qa_report.md` | not in git |
| Run 1 QA report | Same for run 1 | `E:\coding projects\Vidbush\output\alaska\qa_report.md` | not in git |
| Frame review (JSON) | Full findings of the four-area frame review of run 2, with every defect, cause and proposed fix | `E:\coding projects\Vidbush\review_run2\review_run2.json` | not in git |
| Frame review stills | Extracted frames per area (ai_stills, footage, maps, onscreen_text, synth) | `E:\coding projects\Vidbush\review_run2\` | not in git |
| Run 2 log | Detached run output, JSON lines plus yt-dlp progress | `C:\projects\vidbush_tmp\data\_run2.log` and `data\logs\alaska_run2.jsonl` | not in git |
| Ledger | Requirement checklist for run 2 (all items closed) | `C:\projects\vidbush_tmp\.workflow\LEDGER.md` | git-ignored |

The final video to watch: `E:\coding projects\Vidbush\output\alaska_run2\final.mp4` (11:02).
Run 1 video for comparison: `E:\coding projects\Vidbush\output\alaska\final.mp4` (10:47).

---

## 1. Short version

- Run 2 finished end to end with real providers only (0 mock calls): `final.mp4`, 662.3 s, 1920x1080 h264, -14.0 LUFS.
  Cost 2.57 USD (0.233 USD per finished minute). Wall clock about 2 h 46 min.
- Adam's biggest run 1 complaint is fixed: full screen title cards fell from 44 (55 percent of runtime, held 6 to 12 s)
  to 4 (2.9 percent, held 4 s). Text overlays now appear 0.2 to 0.8 s after the word they belong to and mostly
  paraphrase instead of repeating the narration.
- Dog sled teams are now drawn correctly by the AI image model (all 7 team shots).
- Maps still fail about half the time (5 or 6 of 9), but for new reasons: they appear before the place is spoken, a
  title names a place that is not drawn, and labels collide. The fake route geometry of run 1 is gone.
- The big regression: AI stills now fill 65 percent of screen time (run 1: 37 percent). The screen time the cards gave
  up went to AI stills, not to real footage. Sourcing left 21 of 126 shots empty because a relevance filter threw away
  every result. That filter is now fixed (rank instead of drop) but the fix has not run live yet.
- New error classes found by looking at frames: pictures play one shot off their words inside a beat (19 shots),
  labels and boxes that contradict the picture (about 14), split-panel AI collages (7), one fake face of a real person
  (Larry Csonka).
- QA said 90.5 percent of shots are at or above threshold, but the frame review found far more defects. QA samples
  frames about 1.5 s late and its text metric skips cards, map titles and lower thirds, so the number is optimistic.
- Every seat in Adam's 2026-11-11 policy answered a live probe. Narration used direct ElevenLabs (eleven_v4) as
  instructed; Elevate TTS is wired and probed as the fallback.
- Test suite on the merged tip of live-run-2: 1185 passed, 17 skipped, 0 failed.

---

## 2. Timeline of what was done

### 2.1 Merge 1: the two lanes from the day before

- Checked that the local checkout matched GitHub, then merged `review-page` (per-shot review sheet, review export,
  apply fixes, calibration) and `tts-seats` (TTS seats, native Anthropic provider, streaming, seat policy, source
  budgets) into `main`.
- Doc conflicts in DECISIONS, HANDOFF, PROGRESS and PROVIDER_NOTES were resolved by keeping both sides. The tts-seats
  HANDOFF section was renumbered from 9 to 10 (10a to 10f) and every "section 9f" reference became "10f".
- Result: `main` at `ec0dd04`, 965 passed, 17 skipped.

### 2.2 Merge 2: stashed local edits

- The stash held a rewrite of AI image prompts (`safe_prompt`: the editor seat removes named people, brands and
  readable text before the prompt reaches the image model), its test, and a `.gitignore` change.
- Committed as `6287e3e` on `main`. `main` and `live-run-1` both pushed to that commit. 966 passed.

### 2.3 Live Run 2 brief, steps 1 to 3

- Step 1, run 1 state: run 1 had finished. `final.mp4` 646.6 s, QA 72.6 percent at or above threshold 70, 27.4
  percent accepted without a match, cost 2.09 USD (0.193 per minute), 0 mock calls. Its `project.json` still says
  packaging "running" because a later edit pass never marked it done; the video itself is complete.
- Step 2: updated, installed, ran pytest (966 passed, 17 skipped; skips are espeak-ng and Remotion not installed) and
  `doctor` (binaries, faster-whisper and the YouTube runtime ok; every LLM seat false because `.env` used old
  variable names).
- Step 3 checkpoint: Adam gave `VIDBUSH_CONTACT=vidbush@plyon.app`. It was added to `.env` (never printed).
  Commit `da191b2`: the contact User-Agent goes only to Wikimedia hosts, whose policy asks for it, not to every source.
- `YOUTUBE_COOKIES_FROM_BROWSER` was not set: run 1 hit no YouTube bot check (the only bot check came from the
  Library of Congress).

### 2.4 Adam's message of 2026-11-11

Three parts:

1. Contact value (done above).
2. Run 1 feedback: maps often wrong (the "Hurricane Turn Route" example with a random red diagonal labelled 58 miles);
   some AI images bad (the serum run dog team drawn as a man running behind one husky on a leash); far too many title
   cards held 6 to 7 s, wants a few for emphasis and more text over the picture; overlay text too early (5:04 and
   1:50 examples); overlays repeat the narration word for word.
3. Seat policy (section 4) and pasted animation research (HyperFrames, D3/ECharts, MapLibre plus OSRM, Natural Earth,
   Protomaps, SVG/Lottie, Three.js, image-to-video, Remotion, Revideo) with "see which ones would work best for our
   set up and plan accordingly".

Ultracode was switched on at this point, so the work ran as multi-agent workflows.

### 2.5 Trace of run 1 defects (read-only workflow)

Before building, a workflow traced each complaint to code, using run 1's files. Findings that shaped the lanes:

- Overlays: 27 of 33 overlays were verbatim key-phrase callouts. Overlay start always equalled shot start (no anchor
  field), median 2.4 s early, worst 8.7 s early.
- Cards: the documentary profile asked for 12 percent full screen cards plus forced overlay slots; text cards had no
  hold cap.
- Maps: routes were whatever points the director model typed, smoothed into arcs. No roads, railroads or states in
  the base data, so "shaded Alaska/Texas" shaded nothing. Small places (Toksook Bay, Coldfoot) were missing from the
  gazetteer and kept wrong, sign-flipped coordinates.
- AI stills: `must_show` was cut to 4 items before the prompt, so "team" and "sled" fell off. No vision check of AI
  stills against the narration.
- Animation engines: see section 10.

### 2.6 Five fix lanes (workflow, git worktrees on E:)

Five lanes were built in parallel, each in its own git worktree under `E:\coding projects\Vidbush\worktrees\` on a
branch `run2/<lane>`, from `da191b2`. Each lane went build, then adversarial review, then fix of every confirmed
finding. 15 agents, about 74 minutes. Details in section 5.

The first attempt failed because the workflow's built-in worktree isolation refused the `c:` vs `C:` path casing;
the worktrees were created by hand on E: instead (this also follows the "bulk storage on E:" rule).

### 2.7 Merge of the five lanes into live-run-2

Details in section 6. Code conflicts between cards and overlays (7 files) and between maps and overlays (3 files)
were resolved by hand, plus three merge bugs fixed with their own principles. Full suite after the merge: 1117
passed, 17 skipped. Commit `29e8f4c`, pushed.

### 2.8 Environment and live probe

- `.env` and `.env.local` were checked by variable name only (values never printed or compared except by equality):
  every key the seat list needs is present. The image keys in `.env` equal the Elevate and Algrow keys in `.env.local`.
- `OUTPUT_FOLDER=E:/coding projects/Vidbush/output` was added to `.env`, and run 1's output (6.8 GB) was moved there
  from `C:\projects\vidbush_tmp\output\alaska` (C: had 44 GB free). An empty, locked `output\alaska` folder is left
  on C: (held by a file watcher; harmless).
- `python -m vidbush doctor --probe`: every seat answered (table in section 4.2). Wellflow gemini twice returned a
  "model temporarily unavailable" stream event and answered on a re-probe in 21.6 s. Recorded in PROVIDER_NOTES.

### 2.9 Run 2

- Started 2026-11-11 07:49 (local), detached, `python -m vidbush run <alaska script> --id alaska_run2 --brand default`,
  output to `data\_run2.log`. Script: run 1's own copy at
  `E:\coding projects\Vidbush\output\alaska\inputs\05-alaska-size-narration.md`.
- Finished 10:34. No crash, no resume needed. Full numbers in section 7.

### 2.10 Fixes made while run 2 was running (for run 3)

- Sourcing losses (workflow: build, two adversarial reviewers, fix): merged as `3ca4f64`. Section 8.1.
- AI prompt name handling (agent, independent review, redesign): merged as `105812e` plus `0a3cf7f`. Section 8.2.
- Both were merged after run 2 had loaded its code, so run 2 did not use them. They first act on run 3.

### 2.11 After the run

- `vidbush serve` started; http://localhost:8080 opened `alaska_run2` with all 130 shots in the strip, each with
  thumbnail, type badge and score; deliverables and 3 thumbnails rendered (checked in headless Chrome, screenshots in
  `E:\coding projects\Vidbush\logs\serve_alaska_run2*.png`). The server is no longer running (the session ended).
- A four-area frame review workflow (maps, on-screen text, AI stills, real clips) extracted frames from `final.mp4`,
  compared each with the narration at that moment, and a synthesizer verified the key claims on its own frames.
  Section 9.
- Decisions logged, ledger closed, merged worktrees removed, everything pushed.

---

## 3. Current state

### 3.1 Git

| Branch | Commit | Where | Notes |
|---|---|---|---|
| `live-run-2` | `7b8ad4d` (+ this report) | local and GitHub | All run 2 work. Not merged into `main` |
| `main` | `6287e3e` | local and GitHub | Merge 1 plus merge 2 |
| `live-run-1` | `6287e3e` | GitHub | Same as main |
| `run2/seats`, `run2/overlays`, `run2/cards`, `run2/maps`, `run2/ai`, `run2/sourcing`, `run2/aikeep` | various | local only | All merged into live-run-2. Safe to delete |

`main..live-run-2`: 77 files changed, 5,352 insertions, 1,304 deletions.

Commits on live-run-2 since `main`, oldest first:

| Commit | What |
|---|---|
| `da191b2` | Contact User-Agent only to hosts whose policy asks for it |
| `7de52e8`, `e9dceee` | Seats lane and its review fixes |
| `f5584b7`, `9c6db8a` | Overlays lane and its review fixes |
| `5e95937`, `117b901` | Cards lane and its review fixes |
| `068852d`, `a342ef7` | Maps lane and its review fixes |
| `af0e492`, `60b192b` | AI stills lane and its review fixes |
| `45ca404`, `d0b3f11`, `b547482`, `0cb063c`, `d781bbe` | Merges of the five lanes (with hand-resolved conflicts) |
| `29e8f4c` | Merge fixes: one spoken year parser, em dash test reads repo files, probe notes |
| `b332f2f`, `a7ddd24`, `3ca4f64` | Sourcing losses fix, its review fixes, merge |
| `ad1a95e`, `2a277f9`, `105812e` | AI prompt name handling, redesign after review, merge |
| `0a3cf7f` | Removed names deleted in any casing |
| `7b8ad4d` | Decisions: run 2 operations |

### 3.2 Environment

- `.env`: `VIDBUSH_CONTACT` added, `OUTPUT_FOLDER` added. Old variable names (`LLM_PRIMARY_*`, `LLM_WRITER_MODEL`,
  `LLM_FALLBACK_*`, `LLM_CHEAP_MODEL`) are still there. They are harmless now: exact-id seat entries read no pin, and
  a pin that contradicts its seat is ignored with a warning. Removing them is an operator edit.
- `.env.local` is now read by the app (order: `.env`, then `.env.local`, then the process environment; later wins).
- `VIDBUSH_ALLOW_MOCK_FALLBACK=0`, `VIDBUSH_OFFLINE=0`. Mock fallback was never widened.

### 3.3 Storage

| Path | Size | What |
|---|---|---|
| `E:\coding projects\Vidbush\output\alaska` | 6.8 GB | Run 1 project (moved from C:) |
| `E:\coding projects\Vidbush\output\alaska_run2` | 8.7 GB | Run 2 project |
| `E:\coding projects\Vidbush\review_run2\` | small | Frame review output |
| `E:\coding projects\Vidbush\logs\` | small | Full suite logs per lane, probe, serve, screenshots |
| `E:\coding projects\Vidbush\geo_src`, `geo_tools` | | Natural Earth sources and build tools for the bundled map layers |
| `C:\projects\vidbush_tmp\output\alaska` | 0 | Empty, locked folder; delete when nothing holds it |
| `C:\projects\vidbush_tmp\data\source_cache` | | Search and download cache (re-runs reuse it) |

### 3.4 Tests

- After the lane merge: 1117 passed, 17 skipped.
- Sourcing branch alone: 1155 passed. AI names branch alone: 1184 passed.
- Merged tip (all of the above together): 1185 passed, 17 skipped, 0 failed (2026-11-12, 843 s). This is the code run 3 will use.
- The 17 skips are espeak-ng (narration tests) and Remotion (not installed), as before.

### 3.5 Processes

Nothing is running. Run 2 exited normally; `serve` stopped with the session.

---

## 4. Seats (providers per role)

### 4.1 Policy as implemented (config/seats.yaml)

| Role | 1st | 2nd | 3rd |
|---|---|---|---|
| writer (Opus 5.5) | Wellflow cheap key `claude-opus-5-5` | CloseRouter `anthropic/claude-opus-5.5` | Anthropic direct `claude-opus-5-5` |
| director, researcher (Sonnet 5.5) | CloseRouter `anthropic/claude-sonnet-5.5` | OpenLux `claude-sonnet-5-5` | Anthropic direct `claude-sonnet-5-5` |
| editor (Grok 4.7) | CloseRouter `x-ai/grok-4.7` | OpenLux `grok-4.7` | Wellflow (normal key) `grok-4.7` |
| judge, vision, cheap_classifier (Gemini 3.8 Flash) | CloseRouter `google/gemini-3.8-flash` | OpenLux `gemini-3.8-flash` | Wellflow `gemini-3.8-flash` |
| tts | ElevenLabs direct `eleven_v4` (voice Brian) | Elevate TTS `eleven_v4` (voice George) | Algrow TTS `eleven_v4` (voice George) |
| image | Elevate `gpt-image-2.5-fast` | Algrow `gpt-image-2.5` | |
| video, search | none configured | | |

Notes:

- The editor role (prompt rewrites, safe image prompts, requeries) was put on Grok 4.7, as the closest fit to Adam's
  list.
- Adam's normal TTS order (Elevate first, Algrow fallback) is one line to swap back in `config/seats.yaml` once the
  direct ElevenLabs credits run out.
- Every relay streams. Prices are on the role entries (Wellflow cheap 0.30 flat per Mtok, OpenLux Sonnet 0.1765 /
  0.8824, CloseRouter Opus 0.205866 / 1.029332).
- Model ids not verified in advance are resolved from each provider's own `GET /models` list by family and version,
  never guessed.
- An auth, credit or "model not in channel" error now marks that provider and model down for the probe interval
  (`SeatUnavailable`). A one-off error falls back for that request only. A 429 that names a spent balance counts as
  seat-level; a plain rate limit does not.
- One video keeps one narrator: if TTS chunks come from different providers, the odd ones are re-synthesized on the
  serving seat.

### 4.2 Live probe of 2026-11-11

Every entry above answered "ok", including Anthropic direct (it had answered "usage limits until 2026-12-01" during
the build; it works now). Wellflow gemini answered for vision in about 56 s, failed twice for judge and cheap
classifier with `model_unavailable` (temporary), then answered on a re-probe in 21.6 s. During run 2, CloseRouter
returned one 503 `route_unavailable` for the editor role, which was marked down and served by OpenLux.

### 4.3 Elevate TTS request shape (ported from RadioPlay)

`POST /v2/media {type: "tts", prompt, voice_id, model_id}`, then poll `GET /v2/media/{id}?type=tts` every 18 s,
then download `result_url` without auth. Timeout 400 s plus 0.002 s per character, at most 600 s. A concurrency 429 is
waited out, a plain 429 retried 4 times, a task still running at its deadline is resumed (never paid twice). No word
timings come back, so faster-whisper aligns the audio. The probe checks the key through `GET /v2/media/usage` and the
lane through `/v2/media/health`. Adapter: `vidbush/providers/tts_elevate.py`.

---

## 5. The five fix lanes in detail

Every item is a general rule, not a patch for one shot. Each lane's full decision list is in DECISIONS.md under its
"Run 2: ..." section.

### 5.1 Seats lane (`run2/seats`)

- `.env.local` is read; the seat list states the policy in exact order; model ids come from the provider's list.
- Stale model pins in `.env` are ignored with one warning.
- `SeatUnavailable` for seat-level errors (401, 402, 403, out of credit, model not in channel); one shared content
  policy word list so a single refused prompt never marks a whole provider down.
- New Elevate TTS adapter plus one registry line; the model that spoke is logged; each TTS chunk gets the text
  before and after it (`previous_text`, `next_text`), also when one beat is re-narrated.
- Review found 5 issues (429 quota handling, mixed narrators, policy words, probe not checking the key, re-narration
  missing neighbour text); all 5 fixed.

### 5.2 Overlays lane (`run2/overlays`)

- An overlay belongs to a spoken moment, not to a shot. `MotionSpec` gained `anchor` and `start_at`. Text appears
  0.2 s after its anchor word, holds its reading time (at least 1.5 s, 0.3 s per word), moves to the shot that holds
  that moment, or is dropped.
- On-screen text never repeats the narration (4 or more consecutive spoken words, or 60 percent of its content
  words). Rejected text is dropped, never replaced by narration words. The heuristic key-phrase callouts and quote
  cards were deleted.
- The director prompt asks for overlays that add a number, date, place, role, unit conversion or source, or a 2 to
  6 word paraphrase, and to name the words they belong to.
- Overlay slots no longer force text: with nothing honest to show, the shot stays plain.
- The renderer and segment builder honour `start_at`; editor operations (split, merge, resize, re-narrate) keep an
  overlay on its moment or drop it.
- QA gained `overlay_text` stats (verbatim percent, median timing gap, unanchored count) and acceptance checks them.
- Review found 7 issues; all fixed.

### 5.3 Cards lane (`run2/cards`)

- Static text and number cards hold at most 4 s (`card_hold_max`); full screen maps and charts at most 7 s. The
  excess goes to a neighbour or becomes footage of the same beat.
- Documentary mix: full screen cards 5 percent (was 12), animated overlays 37 percent (was 30). Cards at least 75 s
  apart, the hook included.
- A card shows on its cue (its number or key word spoken within its first 1.5 s). Off cue it moves; if it cannot,
  it is demoted and its message rides on footage (title as lower third, number as transparent overlay, map or chart
  as inset).
- Chapter titles became lower thirds; call-to-action beats get no card; a spoken year makes a year card, not a
  count-up.
- QA and acceptance measure card time as a share of runtime and the longest hold.
- Review found 3 issues (year cues, "fifteen hundred" read as a year, missing chart fallback); all fixed.

### 5.4 Maps lane (`run2/maps`)

- A route is drawn only from real geometry between two places the gazetteer resolves. Rail, road and river routes
  snap to bundled Natural Earth layers with a Dijkstra path (new `vidbush/mg/routes.py`); sea and air routes are a
  dashed arc; a route with no named mode tries rail, then road, else is dropped. Model coordinates are never drawn.
- New offline layers in `assets/geo`: 10m railroads (contains the Alaska Railroad; Talkeetna to Hurricane snaps at
  about 55 miles), 10m roads, 50m states and provinces. 13.3 MB in total.
- One shared `verify_map` checks every map a model writes (motion card, inset, editor): markers must be gazetteer
  places inside their own view (sign flips corrected), distances must match the drawn geometry within 25 percent or
  are dropped, shaded regions must exist (states now shade), arrows dropped, source note "Map data: Natural Earth".
- The same map is never shown twice within the inset repeat window.
- Names the script gives to people or animals ("his lead dog, Togo") are never read as places.
- Review found 8 issues (creature names, route endpoints, Georgia state vs country, straight "line" routes, distance
  relabels, repeat via own overlay, editor maps, research anchors); all fixed.
- Known limit: small places missing from Natural Earth (Hurricane, Coldfoot, Deadhorse, Toksook Bay) are dropped, and
  a map title is still free text (see section 9.1).

### 5.5 AI stills lane (`run2/ai`)

- The director keeps counts and relationships in the intent and `must_show` ("a team of dogs pulling a sled with a
  musher riding behind"); the prompt keeps every `must_show` item (the cut to 4 is gone).
- No AI still is used unjudged: generate, judge against intent, `must_show` and the narration, regenerate with a
  "Must clearly show" line up to `judge.ai_retries` (2), best take wins. The judge's reason is context for the
  rewrite only, never prompt or negative-prompt text.
- Subjects whose meaning is readable text (headline, newspaper page, road map, document) are never AI stills; they
  keep their sourced candidate and are flagged `needs_motion_graphic`.
- Spoken years ("nineteen twenty-five") are read as years, so era detection works on TTS scripts.
- The QA report is written before thumbnails so it always matches the `final.mp4` beside it.
- Review found 7 issues; all fixed.

---

## 6. Merge of the lanes: conflicts and fixes

- Doc conflicts (DECISIONS, HANDOFF, ARCHITECTURE): kept both sides.
- Cards vs overlays (acceptance, qa, director, ideas, rules and two tests): both rewrote the heuristic text ideas.
  Overlays removed quote and callout cards of the beat's own words; cards moved chapter titles to lower thirds and
  removed the closing card. Both removals stand, so `text_ideas` is gone. Acceptance keeps both lanes' checks.
- Maps vs overlays (assembler, director, seat): the overlay pool keeps the overlays lane's wider pool and applies the
  maps lane's no-repeat rule to every map in it.
- Merge bug 1: a map inset showed twice within 10 s after merging. Rule: a guarantee is checked after the last step
  that can break it, so the 30 s overlay repeat check now runs again after overlay timing (`drop_repeated_overlays`).
- Merge bug 2: the cards and AI lanes each added a spoken-year parser to `ideas.py` under the same names, and the later
  regex silently replaced the earlier one (13 test failures). Rule: one problem, one implementation; `spoken_years`
  stays.
- Merge bug 3: the em dash test flagged `CLAUDE.local.md`, an auto-generated git-ignored operator file. The test now
  reads only files git tracks or would track.

---

## 7. Run 2 in numbers

### 7.1 Output

- `E:\coding projects\Vidbush\output\alaska_run2\final.mp4`: 662.27 s (11:02), 1920x1080, h264 plus AAC, content
  letterboxed to 1920x892. Mix -14.0 LUFS, true peak -1.4 dB.
- Deliverables: `captions.srt`, `captions.ass`, `chapters.txt` (11 chapters), `description.md` (170 words, 3 titles),
  3 thumbnails ("ALASKA DWARFS TEXAS", "TRUE SIZE REVEALED", "MAPS LIE EXPOSED"), `qa_report.md/json`,
  `cost_report.json`, `shotlist.json`.
- Project status `done`, 13 stages done (research and outline skipped because the input is a finished script).

### 7.2 Wall clock per stage

| Stage | Run 2 (s) | Run 1 (s) | Notes |
|---|---|---|---|
| intake | 4.4 | | |
| script | 0.1 | | |
| narration | 75.1 | 55 | ElevenLabs direct, 11 chunks, 1,841 words, 661.3 s of audio, timings from ElevenLabs |
| shotlist | 441.9 | 140 | 130 shots, 486 ideas (138 from the director seat), 2 off-cue cards moved onto footage. Slower because the director seat writes richer ideas and the new rules run |
| sourcing | 2,725.9 | 3,595 | 677 candidates, 21 shots empty (run 1: 474 and 2) |
| judge | 5,507.3 | 4,184 | 126 shots judged, 114 passed, 98 reworked, 12 no match; 98 requeries; 112 AI stills generated, 29 regenerated for a low score |
| visuals | 0.2 | | |
| motion | 167.9 | about 257 | 44 graphics rendered (overlays included), 0 cached |
| mix | 15.4 | 17 | 5 music sections |
| assembly | 439.4 | 36 | 130 segments rendered, 0 from cache (run 1's 36 s was mostly cache) |
| packaging | 558.9 | | QA frame judging, captions, package, thumbnails |
| total | about 9,936 (2 h 46 min) | | |

### 7.3 Sourcing detail

- Searches used by the end of the run: YouTube 250 of 250 (105 shots), Pexels 150 of 150 (125 shots, 120 queries
  deferred), Wikimedia, Internet Archive and NASA 707 of 800 each (124 shots), Library of Congress 1 of 300 (bot check
  page, skipped for the run).
- The relevance filter dropped results 355 times during sourcing (0 in run 1, where it did not exist yet). This is the
  main reason 21 shots came back empty (section 8.1).
- Prefilter over the whole run: 881 clips checked, 107 rejected (78 burned-in captions, 22 channel logos, 7 people
  talking to camera).
- Downloaded video codecs: 304 h264, 15 vp9, 1 av1, 1 vp8, 1 mpeg4 (YouTube alone: 266 h264, 4 vp9, 1 av1).
- 21 sources under the Standard YouTube Licence (listed in `qa_report.md`); their reuse rights need confirming before
  any upload.

### 7.4 QA (machine)

| Measure | Run 2 | Run 1 |
|---|---|---|
| Shots at or above threshold 70 | 90.5% | 72.6% |
| Accepted without a match (target 1%) | 9.5% (12 shots) | 27.4% |
| Caption coverage and word match | 100% / 100% | 100% / 100% |
| Attributed assets | 37 | 11 |
| NC or ND assets | 0 | 0 |
| Standard YouTube Licence assets | 21 | |
| Mock seats | none | none |
| On-screen text items, verbatim, median timing, unanchored | 30, 0%, 0.2 s, 0 | not measured |
| Shots below threshold on the rendered video | 48 | |
| Flagged shots | 74 | |

### 7.5 Visual mix (share of shots and of screen time)

| Type | Target % | Shots % | Screen time % | Count | Run 1 screen time % |
|---|---|---|---|---|---|
| footage | 20 | 11.5 | 9.6 | 15 | 3.2 |
| footage_overlay | 47.2 | 7.7 | 7.2 | 10 | 3.0 |
| image | 7 | 9.2 | 9.3 | 12 | 0.6 |
| image_overlay | 20.8 | 5.4 | 6.2 | 7 | 0.9 |
| ai_image | 0 | 63.1 | 64.9 | 82 | 37.5 |
| motion_graphic | 5 | 3.1 | 2.9 | 4 | 54.7 |

Caution: 7 shots labelled footage or image are actually AI stills (a converted shot keeps its old type), so real
footage and stills are below the 32.3 percent of screen time this table suggests.

Animation table:

| Measure | Target | Achieved |
|---|---|---|
| Full screen cards (share of shots) | 5% | 3.1% |
| Shots with an overlay | 68% (ceiling) | 30.8% |
| Map and chart insets over the picture | | 6.2% |
| Full screen card time (share of runtime) | at most 10% | 2.9% |
| Longest text or number card hold | at most 4 s | 4 s |

### 7.6 Cost

Total 2.57 USD, 0.233 USD per finished minute, 0 mock calls (run 1: 2.09 USD, 0.193 per minute).

| Item | USD |
|---|---|
| TTS, ElevenLabs direct eleven_v4 | 1.840 |
| Director, CloseRouter Sonnet 5.5 | 0.272 |
| Editor, CloseRouter Grok 4.7 | 0.233 |
| Judge, CloseRouter Gemini 3.8 Flash | 0.144 |
| Vision, CloseRouter Gemini 3.8 Flash | 0.047 |
| Images on the Algrow fallback (2 images) | 0.022 |
| Editor, OpenLux Grok 4.7 (after one CloseRouter 503) | 0.012 |
| Writer, Wellflow cheap Opus 5.5 | 0.0003 |
| Elevate images (about 110) | 0 (subscription) |

---

## 8. Fixes made during run 2 (they first act on run 3)

### 8.1 Sourcing losses (`run2/sourcing`, merged `3ca4f64`)

Cause, found by a read-only investigation that rebuilt the run offline from the search cache:

- 15 of the 21 empty shots lost every result to the relevance filter added late in run 1 (`MIN_RELEVANCE 0.34`, which
  also rejected exactly one in three). YouTube and Pexels hits carry only a title, so a 6-word query needs 2 title
  words. The stemmer missed short plurals ("dogs" vs "dog", "planes" vs "plane", "Alaskan" vs "Alaska").
- 4 shots had results pass the filter but produce no file, silently (the YouTube 720p floor logged nowhere).
- 2 image shots got nothing from Wikimedia, Archive or NASA, and Pexels was out of budget.
- Contributing: image shots held a share of YouTube's video-only budget; deferred queries were never run; Pexels was
  charged for searches never sent (150 used, 54 sent).

Fix, each a rule with tests (20 new tests, then 8 more after two adversarial reviews):

- A relevance filter ranks results, it never removes the last ones. Off-topic results become low-score fillers,
  fetched only when no match gave a file, at most a quarter of the cap, and flagged `off_topic` so a re-run sources the
  shot again. The threshold now accepts one in three.
- Words match in their singular and plain forms, without collapsing onto another common word (plane stays plane, not
  plan).
- A shot shares only the budgets of the sources that can serve its media.
- A search is charged exactly when it is sent (refund on throttle, double charge on the YouTube CC pass fixed).
- Budget left at the end retries only the searches that were never sent, after every shot has had its first search;
  each video source holds back one search per still shot that still needs a frame fallback.
- Every dropped candidate is in the run log with its reason.
- A fallback to video frames is decided on matches, not on fillers.

### 8.2 AI prompt name handling (`run2/aikeep`, merged `105812e`, `0a3cf7f`)

Cause: in run 2 the "rewrite dropped content, original prompt kept" warning fired 27 times. Many were false (names,
pronouns, "single" vs "one"), and the fallback sent the original prompt, names included, to the image model. The
first fix guessed names from capital letters; an independent review showed it still leaked names (McDonald, O'Brien,
JOHN SMITH, a name opening the prompt, a retry hint that asked for the name back).

Redesign. Rule: the editor that removes names is the one that names them.

- The editor returns `{"prompt", "removed": [names, brands, readable text it removed]}`.
- The keep check ignores anything in `removed`; counts and their synonyms are equal (one, single, 1; three, 3);
  hyphen and possessive forms match.
- One retry, which lists only lost words that are not names, and asks to remove any name still present.
- The fallback deletes the removed names from the original prompt (any casing), tidies it, and uses the rewrite if too
  little is left.
- A unicode-aware name heuristic runs only when the editor gives no usable answer, and is logged.
- 68 tests in `tests/test_ai_stills.py`, including a leak matrix of 7 name shapes by 3 editor behaviours.

---

## 9. Frame review of the run 2 render

Four agents each extracted frames from `final.mp4` at the shot times, read the narration at that moment from the word
timings, and judged. A fifth checked the key claims on its own frames. Full detail, including every defect, cause
and file reference: `E:\coding projects\Vidbush\review_run2\review_run2.json`.

The areas overlap, so the counts do not add up: maps 6 of 9 defective; on-screen text 30 of 44 items with at least one
problem; AI stills 31 of 68 checked defective; real footage and stills 17 of 44 shots with a defect.

### 9.1 Maps and charts (9: 1 full screen map, 7 map insets, 1 bar chart inset)

OK: shot_038 (Talkeetna, correct dot, over the real Hurricane Turn train), shot_059 (Anchorage, on its word),
shot_128 (shoreline vs Earth bar chart, correct proportions).

| Time | Shot | Problem |
|---|---|---|
| 02:18 | 034 | "Bethel to Red Devil" inset with a correct river route over summer highway footage, 10 to 15 s before Red Devil is spoken |
| 02:22 | 035 | Second "Red Devil" inset 4 s later, one dot on beige, still before the name is spoken |
| 02:29 | 036 | Full screen map at the right moment but with no route (used up in 034); 17 s of Red Devil maps in a row |
| 07:20 | 090 | Title "Nome to Wrangell-St. Elias" but only Nome is drawn; credit prints over a river label |
| 09:29 | 113 | "Texas, Florida and Alaska": TEXAS and FLORIDA labels overprint, Alaska hidden by the title chip, map over an AI map |
| 10:34 | 125 | Texas map over Inside Passage boat footage, 21 s before Texas is spoken; cluttered river names |

Causes: map insets have no anchor and start at shot start (only text overlays are timed); spacing and novelty guards
apply only to one of the inset paths; the map title is free text never compared with what is drawn; label collision
avoidance covers markers only. Missing story maps: the serum relay (Nenana to Nome) and the Dalton Highway 240 miles.

Versus run 1: fake geometry gone (the 58-mile diagonal, invented curves), holds fine, but the failure rate is about the
same (run 1 about 8 of 16).

### 9.2 On-screen text (44 items: 33 overlays, 1 chart, 4 full screen cards, 7 map titles)

- Fine: 14. Every anchored text overlay lands 0.2 to 0.84 s after its word. The 5:04 callout now reads "Relay: team
  to team, day and night" on "carried it as a relay"; 1:50 no longer shows early text.
- Verbatim repeats: 2 (the 1:09 title card, the 6:40 "Coast Guard icebreaker" subtitle).
- Generic one-word labels on random boxes and arrows: 7 ("Headline" on choir girls, "People" at an empty seat,
  "River", "Train" at a line on a paper map, "Town" around a YouTube subscribe button at 9:23, "Islands" around a
  hillside, a bare "10").
- Labels that name the picture wrongly: 7 ("Anchorage" over empty tundra at 0:30, "Fort Worth / 1950s" over
  Manhattan at 1:13, "USCGC Healy" over HMS Protector at 6:37, "Alaska inset" box over Arizona at 9:36, and others).
- Filler subtitles: 8 ("United States" 5 times, decade labels).
- Early by more than 1 s: 7, all map titles. Late: the 1:09 title card (69 s late, repeats the cold-open line).
- Cut short by the shot end (under 2.1 s on screen): 3.
- QA reported "0 percent verbatim" because its metric skips lower thirds, cards and map titles.

### 9.3 AI stills (82 shots, 65 percent of screen time; 68 checked)

- 37 pass, 31 defects: 14 timing (right picture, one shot early or late; swapped pairs 063/065 and 070/071 in the
  serum run), 6 stub prompts with no subject, 5 wrong depiction, geography or count, 7 split-panel collages, 1
  fabricated face of a real person (Larry Csonka, 9:04), 3 artifacts or bad framing (heads cut by the letterbox), 2
  repeats.
- Fixed: every dog team shot (025, 055, 062, 064, 065, 070, 072) shows a harnessed team with the musher on the
  runners.
- Worst: 5:25 (shot_068) the serum carried on a stretcher by four men with one loose dog, in daylight, under "in the
  dark". Same failure class as run 1's one husky.
- Estimate: about 80 percent of the 82 AI shots could have been real material (46 percent documented subjects with
  archival or news material, 33 percent generic stock scenes); about 21 percent are recreations where AI is fair.
- Causes: picture ideas are not tied to narration words inside a beat; stub fallback intents ("Archival footage of a
  sea, 1920s") become subject-less prompts; the AI judge does not check agent, count or real-person likeness; real
  archival stills under 1000 px wide are rejected; stills are generated at 16:9 but the film is letterboxed to
  2.15:1; "X, then Y" intents produce collages; no place lock (Russian or Greenland villages for Nome and Utqiagvik).

### 9.4 Real footage and stills, and the first two minutes

- 0:00 to 2:00, one frame every 4 s: 20 of 31 strict matches (65 percent), 24 of 31 counting loose fits (77
  percent). Run 1: about 75 percent, but with almost no real footage in its first two minutes.
- 37 real sources used (21 YouTube, 11 Wikimedia, 3 Pexels, 2 Archive), against 11 in run 1. Good where they land:
  Toksook Bay census at 1:40, the Hurricane Turn with its banner at 2:46, Healy and Renda at 6:35, Wrangell at 7:29.
- 17 of 44 footage or image shots have a defect: 8 wrong or off-topic pictures, 2 right pictures 5 to 10 s early, 6
  labels that contradict the picture, 1 clip with a channel's subscribe animation (judge score 0, shipped), 2 low
  resolution (the last shot is soft 2000s camcorder footage).
- None had burned-in subtitles, watermarks, channel logos (except shot_112) or people talking to camera. The
  prefilter is working.
- QA frame sampling is about 1.5 s late, so short shots are judged on the next shot's picture.

---

## 10. Animation research: what fits this setup

Conclusion: the map failures are data and timing problems, not engine problems. No engine switch now.

| Tool | Verdict | Why |
|---|---|---|
| Natural Earth | Adopted | Offline, public domain. 10m railroads and roads plus 50m states now bundled (13.3 MB). Routes snap to it |
| Python renderer (current, `vidbush/mg`) | Keep | Already renders maps, charts, number and title cards, overlays at about 10 s per clip; cached; no extra runtime |
| MapLibre plus Protomaps | Later, optional | Only if Adam wants a styled basemap look (terrain, town labels). Render one MapLibre still per map in headless Chrome and feed it as the static base; keep the Python animation on top. Needs per-region pmtiles on E: |
| OSRM | Later, optional | Road routing only (no rail, river, sea); the public demo server is non-commercial and 1 request per second. Only for road routes Natural Earth cannot connect, cached |
| Remotion / Revideo | No | Not installed; its map draws less than the Python one; Node on this laptop is a v27 alpha; CaseFile saw Remotion starve faster-whisper of CPU |
| HyperFrames | No | No documented native Windows path; Node plus headless Chrome per clip |
| D3 / ECharts | No | Charts were not flagged; the Python chart renderer covers them |
| SVG / Lottie, Three.js, image-to-video | Not now | Belong to cinematic inserts and AI video, not to the map problem. Revisit after the footage share is fixed |

---

## 11. Adam's run 1 complaints: scorecard

| Complaint | Status after run 2 | Evidence |
|---|---|---|
| Too many title cards, 6 to 7 s each | Fixed | 44 cards (55% of runtime) down to 4 (2.9%), text cards hold 4 s. One leftover: the hook title card at 1:09 |
| Overlay text appears too early | Fixed for text, not for maps | Text lands 0.2 to 0.8 s after its word (5:04 and 1:50 fixed). All 7 map titles are early, up to 21 s |
| Overlays repeat the narration | Mostly fixed, new problem | 2 verbatim left. New: generic labels, wrong labels, filler subtitles |
| Bad AI images (dog team) | Improved | All 7 dog team stills correct. The 5:25 stretcher shot is the same failure class. AI share rose to 65% |
| Maps wrong about half the time | Not fixed (different failures) | Fake routes gone; 5 or 6 of 9 still fail on timing, titles and label collisions |

---

## 12. Open issues and known debt

Quality (from the frame review):

1. AI stills fill 65 percent of screen time; real footage and stills under 32 percent.
2. Pictures play one shot off their words inside a beat (19 shots).
3. Map insets are not anchored to their spoken place; titles not checked against what is drawn; label collisions.
4. Random highlight boxes and arrows; lower thirds not checked against the picture; filler subtitles.
5. The hook title card repeats the cold-open line 69 s late (same in run 1).
6. AI collages, letterbox head crops, no place lock, one fake real-person face.
7. A judged score of 0 can still ship when nothing better is found (no score floor).
8. QA samples frames 1.5 s late and its text metric skips cards, map titles and lower thirds, so QA is optimistic.
9. A shot converted to an AI still keeps its footage or image type, so the mix table overstates real footage.

Pipeline and operations:

10. The sourcing and AI-name fixes have not run live yet.
11. YouTube (250) and Pexels (150) search budgets ran out; judge stage is long (5,507 s) because of 98 requeries and
    112 AI generations.
12. 21 Standard YouTube Licence assets need a rights decision before any upload.
13. `live-run-2` is not merged into `main`; seven local `run2/*` branches can be deleted.
14. Stale old variable names in `.env` (harmless).
15. Run 1's `project.json` still says packaging "running".
16. Empty locked `C:\projects\vidbush_tmp\output\alaska` folder.
17. Remotion renderer does not read the new overlay `start_at` (not used today).
18. Small places missing from the bundled gazetteer (Hurricane, Coldfoot, Deadhorse, Toksook Bay) are dropped from maps;
    a public domain GNIS or GeoNames subset would fix that.
19. Elevate TTS works in the probe but has not narrated a full video from Vidbush yet.
20. `docs/PROGRESS.md` and the HANDOFF "known gaps" section were not updated with run 2 results (this report is the
    record).

---

## 13. Recommended next steps, ranked

Each item is a general rule. Sizes: S about half a day, M about a day of agent work.

1. **Real footage first (M).** Run 3 with the sourcing fix live; raise the YouTube and Pexels budgets; accept public
   domain archival stills down to about 500 px wide (framed on a blurred background); judge real candidates against
   the narration in their window, not the over-specific intent; prefer any real candidate scoring 55 or more over AI;
   stop adding the video topic to fallback queries; set `shot.type` to `ai_image` when the pick is AI. Target: real
   footage and stills above 60 percent of screen time, AI below 35 percent.
2. **Every picture and map follows the word it illustrates (M).** Give picture ideas a narration anchor as overlays
   have; the assembler penalises distance from the anchor word to the slot; map and chart insets are timed like text
   overlays (anchor on the first place named, move or drop); drop the hook title card when its text is the cold-open
   line. Fixes the 19 one-shot-off pictures, 7 early map titles and the 1:09 card.
3. **No weak fill (S to M).** Build AI prompts from the beat's agent, action, object, place and era, never from a
   stub; reject real candidates below about 50 or with subscribe, channel or watermark notes; the AI judge checks
   agent, count and setting against the narration and fails collages and real-person likenesses; generate at the
   letterboxed aspect; one clause per prompt; fix QA frame sampling. Target: "accepted without a match" near 1
   percent, and QA that agrees with the frames.
4. **Labels only claim what the picture shows (S).** Delete the random box and arrow options; place and person lower
   thirds only when the name is in the shot's `must_show`; no "United States" or decade filler; one label per name per
   minute; text stays at least 2.5 s or moves; QA measures cards, map titles and lower thirds too.
5. **One map per change of place (S to M).** Spacing and novelty guards on every map overlay; rebuild the title from
   the places actually drawn; one collision set for title, credit and all labels; a state locator on close-up maps;
   the director plans the story maps it skipped (serum relay, Dalton Highway).

Then, smaller:

6. Merge `live-run-2` into `main` once Adam has watched run 2 (or after run 3), and delete the `run2/*` branches.
7. Extend the offline gazetteer with a public domain place list so small towns resolve.
8. Decide the Standard YouTube Licence policy (keep for fair use, or prefer Creative Commons and stock).
9. When ElevenLabs credits run out, swap the TTS order back to Elevate first, Algrow second (one line).
10. Optional look upgrade: MapLibre basemap still per map (section 10), only after items 2 and 5.

---

## 14. Adam's review of run 2 (asked for in the brief)

Video: `E:\coding projects\Vidbush\output\alaska_run2\final.mp4`. Or run `python -m vidbush serve` and open
http://localhost:8080, project `alaska_run2` (the shot strip lets you click to any shot).

1. Watch 0:00 to 2:00 and judge only whether each clip matches the narration.
2. Skim to the end. Note timestamps of burned-in subtitles, watermarks, channel logos and people talking to camera.
3. Note any map or chart insets that stay too long or come too often.
4. Write one sentence comparing the feel to your Vidrush video.
5. Do not judge the voice or the script.

Timestamps worth a look first: 5:25 (serum stretcher), 9:22 (a "Town" box around a subscribe button), 1:09 (hook title
card repeated), 2:18 to 2:36 (three Red Devil maps), 4:58 to 5:47 (swapped picture pairs), 9:04 (fake Csonka face),
7:20 (map title naming a place it does not draw), 10:34 (Texas map over boat footage).

---

## 15. Decisions needed from Adam before the next run

- Go ahead with run 3 (about 2.60 USD and 2.5 to 3 hours) after items 1 to 3 of section 13, or after all five?
- Search budgets: how far to raise YouTube and Pexels (more searches mean more wall clock and more Standard YouTube
  Licence clips)?
- Standard YouTube Licence clips: allowed (fair use) or avoided?
- Real people: never generate an AI likeness of a named real person (recommended), always use a real photo or a
  context shot with a lower third?
- Merge `live-run-2` into `main` now, or after run 3?
- Keep direct ElevenLabs for narration until its credits run out, then Elevate first?

---

## Appendix A. Where the run 2 rules live in code

| Area | Files |
|---|---|
| Seats and providers | `config/seats.yaml`, `vidbush/config.py`, `vidbush/providers/seats.py`, `providers/base.py`, `providers/model_resolve.py`, `providers/tts_elevate.py`, `providers/probe.py`, `vidbush/live_probe.py` |
| Narration | `vidbush/audio/narration.py` |
| Overlays and cards | `vidbush/shots/ideas.py`, `shots/rules.py` (`time_overlays`, `cap_card_holds`, `cards_on_cue`, `space_cards`, `demote_repeating_cards`, `drop_repeated_overlays`), `shots/director.py`, `shots/assembler.py`, `vidbush/models.py` (`anchor`, `start_at`, `card_hold_max`) |
| Maps | `vidbush/mg/routes.py`, `mg/mapdraw.py`, `mg/geo.py`, `mg/builders.py`, `shots/seat.py` (`verify_map`), `assets/geo/` |
| AI stills | `vidbush/visuals/ai_images.py` (`safe_prompt`, keep check, `without_names`), `vidbush/judge/director_loop.py` (`add_ai`), `judge/judge.py` |
| Sourcing | `vidbush/source/registry.py`, `source/base.py`, `source/throttle.py`, `source/youtube.py` |
| QA and acceptance | `vidbush/judge/qa.py`, `vidbush/acceptance.py` |
| Render | `vidbush/mg/renderer.py`, `vidbush/render/segment.py` |

## Appendix B. Logs

| Log | Path |
|---|---|
| Run 2 console | `C:\projects\vidbush_tmp\data\_run2.log` |
| Run 2 structured | `C:\projects\vidbush_tmp\data\logs\alaska_run2.jsonl` |
| Live probe | `E:\coding projects\Vidbush\logs\probe1.log` |
| Full suites per lane and after merges | `E:\coding projects\Vidbush\logs\*_full_suite*.log` |
| Full suite on the merged tip | `E:\coding projects\Vidbush\logs\live_run2_tip_full_suite.log` |
| Serve and screenshots | `E:\coding projects\Vidbush\logs\serve.log`, `serve_alaska_run2.png`, `serve_alaska_run2_strip.png` |
