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
