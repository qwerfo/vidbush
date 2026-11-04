# Progress log

One line per milestone: what works, what failed, what is next.

- Tier 0: repo, contracts (vidbush/models.py, docs/schemas), config loader, provider layer with seats, retries, fallback, down probing, cost tracking and mock mode, job runner with resume, SQLite job table, CLI skeleton. Next: Tier 1 to 4 modules in parallel against the contracts.
- Tier 1 to 3 built by 9 parallel engineers against the contracts (437 unit tests passing). Demo (A1) passes every acceptance check in mock mode on the first full run: 4.3 min, 43 shots, 5 shot types, maps, chart, title cards, -14.0 LUFS, -1.1 dBTP.
- A2 (script, Hoover Dam) and A3 (voiceover, Erie Canal) initially missed a chart: the director now guarantees one when chartable numbers exist, parses spelled-out numbers, and labels bars by sentence subject. Both pass.
- A8: injected failure at judge, resume skipped intake to sourcing and completed. A5: revise mapped to retime_section [0,30] x0.7 plus change_type shot_007 ai_image, applied, 8 of 47 segments re-rendered. A6: learn-style found 12 shots, 2.39 letterbox, captions at band 0.48. A4: API create, run, swap re-rendered 1 of 7 segments; UI checked in headless Chromium with no console errors.
- A9: Docker daemon available. deb.debian.org is blocked by the sandbox, so the image moved to an ubuntu:24.04 base; Docker Hub rate limited the pull (429), retrying.
- Five-angle review (spec, real providers, editor and cache, audio and video, output quality) found 63 issues. Five engineers fixed them in disjoint file groups: mock fallback only when a role has no real provider (else the stage fails and resume retries), 429 backoff with Retry-After, cost estimates recorded, image provider order kept, default ElevenLabs voice, path rebasing for moved projects, persisted edit force sets with stage cascade, per-project lock, top10 profile switch and countdown recap numbers, content-based cache fingerprints, music bed without pumping, per-scene director calls, source throttling and search caching, research bounds, deno for yt-dlp. 622 unit tests pass.
- Final clean verification (fresh outputs): A7 integration test passes (second assembly renders 0 segments). A8 resume after injected failure at judge skips intake to sourcing. A1 demo 257.9 s, 42 shots, -14.0 LUFS, -1.1 dBTP. A2 239.9 s, 41 shots. A3 235.9 s, 40 shots. A5 revise: retime_section then change_type, 6 of 45 segments re-rendered, still passes. A6 writes the profile. A9 image rebuilt with deno and served on 8080. A10 guarded by a test. Top 10 smoke run: profile switch, ten number cards, recap grid #10 to #1 with #1 highlighted.
- Windows fixes: Windows-safe pid check in the project lock (os.kill(pid, 0) sends CTRL_C_EVENT there) and a __main__ guard for spawned motion graphic workers.
- Parallel2 (branch mix-director-batch), in four parts:
  - Part A: animation is split into full_screen_cards 12 and animated_overlays 30. Map and chart overlays render as inset panels, and free full screen cards are capped at one per 45 s outside the hook and recap.
  - Part B: every director call carries the script summary, the previous two chunks' ideas, the anchors so far and the next beat.
  - Part C: a writer or director heuristic fallback while a real provider is configured is listed in QA by scene id and sets needs_review.
  - Part D: `batch` CLI with a per-run cost cap, batch_report.md, pause and resume, and a Queue page.
- Tests: 610 passed and 13 skipped before; 658 passed and 13 skipped after, including the slow demo test. The 13 skips are 12 espeak-ng tests (not installed here) and 1 Remotion test.
- A1 demo: 262.9 s, 54 shots. Shot mix: footage 20.4, footage_overlay 42.6, image 9.3, image_overlay 18.5, full screen cards 9.3. Overlays on 61.1 percent of shots, map and chart insets on 18.5 percent. The second assembly rendered 0 segments.
- A2 and A3 pass every acceptance check: cards 9.8 and 10.4 percent, overlays 58.8 and 60.4 percent.
- Done. Read docs/HANDOFF.md.
