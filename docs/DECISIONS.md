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
