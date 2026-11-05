# Decisions

One line each: the decision and why.

- Sandbox blocks Wikipedia, YouTube, Archive, LoC, NASA, OpenAI-style APIs and ElevenLabs. All adapters are built and tested against canned responses; the demo runs in mock mode with a bundled local footage library.
- Added a `local` candidate source (operator footage library folder with sidecar tags). It doubles as the offline fixture source and is useful for owned stock.
- Installed espeak-ng so mock narration is intelligible speech, not tones. Tone bursts remain the last fallback.
- Bundled Inter (OFL) as the one font family, and Natural Earth 110m and 50m countries, 50m lakes and rivers, and a 7342-place gazetteer from Natural Earth 10m populated places for offline maps and geocoding.
- Python 3.13 is what the sandbox has; code targets 3.11+.
- Every LLM caller has a deterministic heuristic fallback, so mock mode works for any input, not only the demo fixtures.
- Director works in two layers: per-beat visual ideas (LLM or heuristic), then a deterministic pacing engine assigns typed, timed slots. This keeps the visual mix and pacing exact regardless of model quality.
- Captions are burned per segment (window of the global caption list) so the segment cache key includes the caption slice, as the spec requires.
- Transitions are hard cuts plus a short fade through black at some scene starts, rendered inside the segment, so segments can be concatenated with stream copy and cached independently.
- Built-in Python motion graphics renderer is the default (about 30 to 60 fps at 1080p, maps pre-render static layers). The Remotion project in renderer/ renders the same MotionSpec and is selectable with render.renderer: remotion; it falls back to Python on any failure.
- Visual mix percentages are measured by shot count (QA also reports screen time share), because motion graphics are longer than footage and a time-based mix would starve footage.
- Music bed level: bed_gain_db is the level under speech relative to the -14 LUFS target; the bed rises by ducking_depth_db in pauses of 0.8 s or more. The ducking key is a smoothed, gap-merged speech envelope, so the bed holds one level through sentences with no per-word pumping.
- Loudness uses ebur128 measurement plus a linear gain and a true peak limiter only when needed (same target as two-pass loudnorm linear mode, faster, verified after the pass).
- Chunk stitching in narration uses a sample-exact numpy 40 ms linear crossfade so word timings shift exactly.
- The director guarantees one data chart per video whenever any beat has two or more comparable numbers, and parses spelled-out numbers ("one hundred dollars").
- Docker image is based on ubuntu:24.04 (Python 3.12, FFmpeg 6, espeak-ng from the Ubuntu archive).
- Built UI (ui/dist) is committed so Windows and Docker users do not need Node to run Vidbush.

## Live run 1 (2026-11-04, Windows laptop)

- Kept the existing FFmpeg (gyan.dev build in C:\FFmpeg) and Git instead of installing second copies with winget; installed Python 3.11 and Deno with winget.
- numpy pinned per Python version (2.5.3 on 3.12+, 2.4.6 on 3.11) because the 3.13 sandbox pin had no 3.11 wheel and the project promises 3.11+.
- Tests load settings with use_dotenv=False, so the operator's keys can never turn a unit test into a live, billed call.
- Fallback text and vision seats use WELLFLOW_API_KEY, not WELLFLOW_CHEAP: the cheap key's channel refused every model in the seat map (HTTP 403 model_not_in_channel).
- Editor and vision roles got their own fallback entries (LLM_FALLBACK_CHEAP_MODEL, VISION_FALLBACK_*), so a CloseRouter Gemini outage does not stop the judge.
- Writer on anthropic/claude-opus-5, researcher and director on anthropic/claude-sonnet-5, editor, judge, vision and cheap_classifier on google/gemini-3.8-flash, all on CloseRouter; fallback qwen3.8-max and gemini-3.8-flash on Wellflow.
- Gemini 3.8 Flash on CloseRouter passed a real vision probe (JSON scores for a JPEG), so the Wellflow Gemini is the fallback, not the primary.
- No max_tokens sent: CloseRouter returned 3,545 tokens with finish_reason stop for a long request. options.max_tokens exists for relays that truncate.
- LLM timeout raised from 180 s to 600 s for every role (writer calls on a slow relay ran past 180 s in RadioPlay).
- Calls without usage.cost are priced from the provider's own /models price list rather than a flat per 1k token guess.
- Elevate and Algrow are async job APIs, so they got their own provider types (elevate, algrow) instead of being forced through the OpenAI images shape.
- Image model on Elevate is gpt-image-2.5-fast (Elevate's id for GPT Image 2.5; 4 credits, subscription, recorded as 0 USD); Algrow fallback gpt-image-2.5 at 0.01085 USD. Both under the 3 cent warn level.
- Elevate images get a 5 percent edge trim to remove the corner sparkle mark (provider option edge_trim, any provider can use it).
- The gpt-image price table only prices gpt-image-1 and gpt-image-1-mini; newer gpt-image ids use the stated cost_per_image instead of being priced as gpt-image-1.
- TTS model_preference eleven_v4 then eleven_multilingual_v2; a refused auto model switches to v2 for the whole run so voices never mix.
- Brand voice kept as the premade Brian (nPczCjzI2devNBz1zQrb): the voice is not under test in this run, and a premade voice exists on every account.
- Search seat left empty: no Tavily key exists and script mode needs no research.
- Font features are dropped when Pillow lacks libraqm (all Windows wheels), so charts and number cards render instead of degrading to text overlays.
- Every provider request sends a browser-like User-Agent (Cloudflare error 1010 on Elevate otherwise).
- doctor --probe now probes every configured provider of a role, not only the first, so a dead fallback shows before a run needs it.
- Pexels key added (Adam supplied a full key file and said to use it); Tavily and Pixabay still absent, so search stays mock and Pixabay stays off.
- Git identity set repo-local to the repo owner (qwerfo) because the laptop had none; no global git config was changed.
- The run is started as a detached process logging to data/<id>_run1.log, because a tool-managed background job is cut off after 10 minutes.
- YouTube format_sort prefers h264 after resolution: AV1 decoded correctly but OpenCV scene detection took 27 to 37 s per clip against 1.4 to 3 s for h264, and 55 of 175 cached clips were AV1. Resolution is never traded for codec.
- Code review fixes went in during the render; the running process keeps the code it loaded, so they apply from the next resume.
- Wikimedia 429s left as they are: its User-Agent policy wants a contact address (VIDBUSH_CONTACT), and Adam's email is not sent to a third party without his say.
- AI image negatives (text, logos, chyrons, the likeness of a named real person) travel as a prose "Do not include" sentence for providers with no negative field; before this fix they were silently dropped, and the first render had a fake TV interview of a real person and stills with invented headlines.
- Shots whose AI stills carry baked text, a fake logo or a real person's likeness are regenerated with the regenerate_ai edit op (segment re-render only), not by re-running the whole project.

## Footage quality and YouTube (parallel lane)

- Footage quality work (parallel1) is developed on the session branch claude/footage-quality-youtube-3eojov and the same commits are pushed to footage-quality, the branch name Adam asked for.
- Prefilter OCR uses rapidocr-onnxruntime 1.2.3: its three onnx models (det, cls, rec) ship inside the wheel, verified by installing and listing the package files, so nothing downloads at runtime.
- Prefilter face check uses the OpenCV haar cascades. mediapipe 1.2.2 (the current wheel) was installed and checked: it has no solutions module and no bundled face model (the tasks API downloads a .tflite), so it was rejected. opencv-python-headless 5.0 no longer ships cv2.data haar files; 4.14.0.94 does (verified), so OpenCV is pinned to 4.14.0.94, with opencv-python at the same version because rapidocr requires it.
- Every new threshold lives in config (judge.prefilter, judge.*_detail, sources.youtube_*, cost.vision_per_image_high_usd) and the brand gets licence_policy.prefer_creative_commons; all new config fields were added in one commit first so the four parts could be built in parallel without config conflicts.
- Prefilter: one entry point (_prefilter_and_score) before every vision call in the director loop (first pass, requery, AI retries). A reject gets score 0 and reject_reason "<label> (prefilter: detail)", which selection already treats as a hard reject.
- Prefilter skips operator and generated media (local library, uploads, AI stills and clips, motion graphics); logo and face checks run on video only, since a still has no temporal evidence and a still portrait is a normal documentary subject.
- Subtitle rule: OCR boxes below text_band_top are grouped into rows, then into runs of horizontally close boxes (gap at most 1.5 text heights); only a run at least text_min_width wide counts, so corner bugs and timestamps on one row never merge into a subtitle.
- Logo rule: a textured, pixel-stable blob inside a corner box, not touching the sides facing the centre, with changing pixels around it; static clips (75 percent of the centre stable), letterbox bars, flat skies and dark corners never count.
- Face rule: haar frontal cascade with minSize derived from face_min_area, so crowds and small faces are never searched; profiles pass because the frontal cascade does not fire on them.
- A prefilter dependency that fails to load (import or model build) disables only its check, with one warning per process.
- YouTube licence policy reaches the adapter as optional search hints declared by the adapter, so the adapter interface and other sources are unchanged. The CC pass uses sp=EgIwAQ%3D%3D (checked against the installed yt-dlp YoutubeSearchURLIE), is capped at 10 items, and a failed CC pass never drops the normal pass.
- standard_youtube_licence is a flag on the candidate and shot, counted in QAReport.standard_youtube_assets and its own qa_report.md section. Flag only, never a block.
- YouTube windows: with captions or chapters, the window (shot x4, 30 to 75 s, stepped by half a window) with the best token overlap wins (must_show weight 2, intent 1, queries 0.5); else up to youtube_windows windows of 15 to 30 s at even fractions between 5 and 95 percent. Each window is its own cached download with 10 s padding, and at most 4 clips per video in total.
- Window files are named youtube_<id>_<true start seconds>; the true start subtracts the keyframe lead measured from the probed file duration, because stream-copied section downloads begin at the keyframe before the requested start. Re-encoding at cuts was rejected as too slow.
- yt-dlp gets a private temporary copy of the cookie file per call, because yt-dlp rewrites its cookie file in place on close and parallel downloads would corrupt it; the operator's cookies.txt is never modified.
- Downloads of different videos run in parallel up to youtube_download_workers with a per-URL lock; searches, info extraction and caption fetches stay on the serial 0.5 per second slot. Search usage per run goes to work/source_notes.json so qa_report shows "YouTube searches used: X of N" and a warning when a budget ran out, across resume.
- Segment dedupe: same segment means same asset, video media, and ranges that overlap or start within 3 s on the source timeline. It is banned within a scene and for adjacent non motion graphic shots (a graphic between two shots does not reset adjacency); in another scene it costs 40 points. The ban is applied before the threshold split: an allowed below-threshold candidate or a fresh AI still beats a banned repeat, and a forced repeat is flagged segment_repeat and no_match_accepted.
- Editor re-sourcing of one shot runs enforce_segment_rule against its neighbours and scene, and in-point variation of repeated video skips starts that would recreate a banned segment.
- Judge detail is a per image list on Seats.vision_json defaulting to low, so only the judge pays for high detail; auto is priced as high in the missing-cost estimate. All vision images are downscaled to fit 640 px before sending so the flat per image prices hold for full-size AI stills too.
- vision_per_image_high_usd default is 0.010 (5x low): a 640x360 frame at high detail is 2 tiles, 85 + 170 x 2 = 425 tokens against 85 at low.
- max_tokens is read from provider or role options in seats.yaml and never sent unless set to a positive int; a 400 pointing at max_completion_tokens teaches a rename quirk. The SPEC line now says no cap by default.
- LLM timeouts resolve per role (writer 600 s, others 180 s); a caller asking for longer wins, and options.timeout_s replaces the role default outright.
- A reply with finish_reason length or max_tokens is continued once (partial as an assistant message plus a continue prompt, no response_format); a second truncation or unparseable joined JSON raises RefusalError, which fails over to the next provider without marking the first one down. Usage of both calls is summed.

## Visual mix, director context and batch queue (parallel lane)

- Visual mix: animation is split into `full_screen_cards` and `animated_overlays` keys inside `visual_mix`. Overlays are counted on top of the shot they decorate, so they are added to footage_overlay and image_overlay in the picture's footage to still ratio (41 to 17). The documentary default becomes footage 20, footage_overlay 42.2, image 7, image_overlay 18.8, motion_graphic 12: the reference's raw shares stay exact and the total stays 100.
- Old profiles that use the `motion_graphic` key read it as `full_screen_cards`, so learned and older profiles keep working. The learner now writes `full_screen_cards` with `animated_overlays` 0, because a classifier already counts the overlays it sees as overlay shots.
- The top10 profile uses 20 full screen cards and 22 overlays (42 in total). Its number cards are full screen by design and are always kept.
- The 45 s card spacing is enforced twice: as a planner penalty, and in a final `space_cards` pass after the chart guarantee. Structural cards (number cards, title card, recap) are never removed and count against free cards near them. When the cap and the mix disagree, the cap wins.
- A map or chart used as an overlay renders as an inset panel (45 percent of the frame width, upper right, larger type). Insets always use the Python renderer, even when Remotion is configured, because Remotion has no inset composition.
- A free map or chart card removed by the spacing cap keeps its content as an inset on footage of its beat, so the information stays on screen.
- An inset map is offered only when it shows a place that no map showed in the last 60 s. Without that limit the Panama scripts put a map on about 20 of 70 shots.
- Insets are kept at least 15 s apart, and the same map or chart is not shown again within 3 minutes. An inset is information the viewer has to read, and the first demo render put one on 43 percent of shots, back to back in the opening.
- MG_RENDER_VERSION is now mg-3, because chart value labels now always fit inside the plot and cached charts must re-render.
- Acceptance adds three checks: cards_sparing (full screen cards at most 25 percent of shots), overlays_on_picture, and inset_map_or_chart.
- Director chunks now run one after another instead of four in parallel, because each call needs the ideas chosen before it. This costs wall time on long scripts: about one call per 8 beats, in sequence.
- The script summary comes from the director seat (task `script_summary`). It is cached in work/script_summary.json, keyed by a hash of the script, so a re-run of the shot list does not pay for it again. Its fallback is the title plus the first sentence of every scene.
- The visual anchors so far are the research anchors that the chosen ideas mention. Script and voiceover projects have no research, so they get the recent must_show subjects instead.
- The director may now propose map and chart overlays. These get the same coordinate and fact checks as motion ideas and render as insets. An inset map must draw a marker, a route or an arrow.
- Heuristic fallback notes are recorded through `seats.note_fallback`, and only when the role has a configured real provider, because mock mode is already reported as mock. Each stage writes its notes to work/fallbacks.json and replaces its own earlier entries, so a re-run that succeeds clears them.
- A director chunk counts as a fallback when the call fails or returns no valid idea. The script summary fallback is reported with scene id "all". A writer section counts when its beats come from fact claims.
- Batch cost cap is checked after each stage against the project's cost_report total (which includes earlier runs), so a run stops at the next stage boundary, never mid stage; stopping inside a stage would leave half written outputs.
- A capped project ends as needs_review with project.status_note and work/cost_cap.json; the batch shows it as capped. No new project status value, so the UI, API and final status logic stay unchanged.
- No cap check after the last stage: the video is finished by then, so stopping saves nothing.
- A batch runs as one worker job (kind batch, project_id batch:<id>) that runs its projects in sequence through jobs_handler.run_pipeline, so batches reuse the job table's crash recovery and the API worker's exact pipeline path.
- Batch pause is a flag in state.json read between projects; the running project always finishes. A per-item resume uses the existing stage resume, so an interrupted batch never recreates a project it already started.
- Batch project ids come from file stems like run (unique suffix _2, _3 within a batch) and replace an existing project of the same id, matching run.
- Batch CLI uses flat names (batch, batch-status, batch-pause, batch-resume) so `batch <folder>` stays a single positional command.

## TTS seats, Anthropic provider, streaming, seat policy, source budgets (parallel1, branch tts-seats)

- Algrow TTS is its own provider type algrow_tts in vidbush/providers/tts_algrow.py, ported from RadioPlay's gateway/adapters/algrow_tts.py (submit, poll, download), because an async job API does not fit the ElevenLabs request shape.
- TTS adapters register in one dict in seats.py (TTS_TYPES, one line per type), so the laptop lane's elevate_tts adds one line and the two branches merge without touching the same lines.
- A tts entry whose type is not registered is skipped with one warning per process, because a seat list may name a type another branch is still adding and a run must not fail for it.
- Algrow is a subscription: calls are recorded as 0 USD and the reported credit count goes into a new optional CostEntry.credits field, so spend stays visible without inventing a dollar price.
- Short Algrow scripts are padded with break tags to its 200 character floor, as RadioPlay does, because Algrow answers HTTP 400 below it; Vidbush chunks are whole scenes, so this only touches tiny scenes and single beat re-narrations.
- Voice ids live per provider in brand voice.voice_ids (name, then type) with voice.voice_id as the one default, because the same voice has different ids on different TTS services.
- Alignment routing: a TTS chunk without usable timings (none returned, drifting, mock, or a seat with force_align: true) is aligned against the script with the Aligner that narration_from_voiceover uses, inside the chunk's exact stitched window rather than across the whole file, because the window offsets are known exactly and only narrow the search.
- force_align is a tts provider option read through Seats.tts_option, because unreliable timings are a property of one provider, not of a brand or a run.
- The first narration chunk is synthesized alone before the rest run in parallel, so a refused model switches the run (sticky in each adapter) before any other chunk exists and a video never mixes models.
- A narration whose chunks came from more than one tts provider is logged as a warning, not redone: redoing would ask the same failing seat order again.
- The native Anthropic adapter speaks the Messages API over the repo's own requests layer, not the anthropic SDK, because every provider shares one retry, 429, fallback and cost path in ProviderClient, and the SDK's own retries and exception classes would duplicate and bypass it.
- The Anthropic adapter never sends temperature (current Claude models reject sampling parameters with a 400); options.send_temperature exists for older models.
- max_tokens is sent on every Anthropic call (64000 streamed, 16000 not, or options.max_tokens) because the Messages API requires it; a 400 naming a lower limit teaches that limit for the model.
- Anthropic JSON is asked for in the system prompt and read with the shared loose parser, because the Messages API has no json_object mode and structured outputs would tie every caller to a schema.
- The Anthropic price table is built in (USD per million, cache reads and 5 minute writes at their own rates) with options.price_per_mtok as the override, because the direct API returns no cost and has no price list endpoint.
- LLM and TTS type registries map a type to its module and look the function up at call time, so a test or a hot fix that replaces llm.chat still takes effect.
