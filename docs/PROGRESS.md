# Progress log

One line per milestone: what works, what failed, what is next.

- Tier 0: repo, contracts (vidbush/models.py, docs/schemas), config loader, provider layer with seats, retries, fallback, down probing, cost tracking and mock mode, job runner with resume, SQLite job table, CLI skeleton. Next: Tier 1 to 4 modules in parallel against the contracts.
- Tier 1 to 3 built by 9 parallel engineers against the contracts (437 unit tests passing). Demo (A1) passes every acceptance check in mock mode on the first full run: 4.3 min, 43 shots, 5 shot types, maps, chart, title cards, -14.0 LUFS, -1.1 dBTP.
- A2 (script, Hoover Dam) and A3 (voiceover, Erie Canal) initially missed a chart: the director now guarantees one when chartable numbers exist, parses spelled-out numbers, and labels bars by sentence subject. Both pass.
- A8: injected failure at judge, resume skipped intake to sourcing and completed. A5: revise mapped to retime_section [0,30] x0.7 plus change_type shot_007 ai_image, applied, 8 of 47 segments re-rendered. A6: learn-style found 12 shots, 2.39 letterbox, captions at band 0.48. A4: API create, run, swap re-rendered 1 of 7 segments; UI checked in headless Chromium with no console errors.
- A9: Docker daemon available. deb.debian.org is blocked by the sandbox, so the image moved to an ubuntu:24.04 base; Docker Hub rate limited the pull (429), retrying.
- Next: five-angle review (spec, real providers, editor and cache, audio and video, output quality), then fixes, final clean runs, HANDOFF.
