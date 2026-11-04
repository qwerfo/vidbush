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
- Visual mix: animation is split into `full_screen_cards` and `animated_overlays` keys inside `visual_mix`. Overlays are counted on top of the shot they decorate, so they are added to footage_overlay and image_overlay in the picture's footage to still ratio (41 to 17). The documentary default becomes footage 20, footage_overlay 42.2, image 7, image_overlay 18.8, motion_graphic 12: the reference's raw shares stay exact and the total stays 100.
- Old profiles that use the `motion_graphic` key read it as `full_screen_cards`, so learned and older profiles keep working. The learner now writes `full_screen_cards` with `animated_overlays` 0, because a classifier already counts the overlays it sees as overlay shots.
- The top10 profile uses 20 full screen cards and 22 overlays (42 in total). Its number cards are full screen by design and are always kept.
- The 45 s card spacing is enforced twice: as a planner penalty, and in a final `space_cards` pass after the chart guarantee. Structural cards (number cards, title card, recap) are never removed and count against free cards near them. When the cap and the mix disagree, the cap wins.
- A map or chart used as an overlay renders as an inset panel (45 percent of the frame width, upper right, larger type). Insets always use the Python renderer, even when Remotion is configured, because Remotion has no inset composition.
- A free map or chart card removed by the spacing cap keeps its content as an inset on footage of its beat, so the information stays on screen.
- An inset map is offered only when it shows a place that no map showed in the last 60 s. Without that limit the Panama scripts put a map on about 20 of 70 shots.
- MG_RENDER_VERSION is now mg-3, because chart value labels now always fit inside the plot and cached charts must re-render.
- Acceptance adds three checks: cards_sparing (full screen cards at most 25 percent of shots), overlays_on_picture, and inset_map_or_chart.
