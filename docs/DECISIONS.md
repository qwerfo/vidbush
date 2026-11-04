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
