# Vidbush

Self-hosted long-form faceless YouTube video production. Give it a brief, a
script, or a finished voiceover. It produces a publish-ready 6 to 60 minute
documentary or Top 10 video: research, script, narration, web-sourced
footage, AI images, motion graphics (maps, charts, number cards, titles,
lower thirds, callouts), captions, a ducked music bed, chapters, a
description with attribution, and three thumbnail options. A shot-level
editor and a chat revision loop let you fix any shot afterwards, and only
that shot re-renders.

Start here: [docs/HANDOFF.md](docs/HANDOFF.md) (setup on Windows 11 and on a
Linux VPS with Docker, seats, costs, known gaps).

```bash
python -m vidbush demo                    # 4 minute documentary from fixtures/demo_brief.md
python -m vidbush run my_script.md        # script mode
python -m vidbush run narration.mp3       # voiceover mode (pairs with narration.md if present)
python -m vidbush serve                   # UI and API on http://localhost:8080
python -m vidbush revise demo "make the first 30 seconds faster and replace shot 7 with an AI image"
python -m vidbush swap demo 12 path/to/my_clip.mp4
python -m vidbush learn-style reference.mp4 my_style
python -m vidbush resume demo
python -m vidbush doctor
```

With no keys in `.env`, every seat runs in mock mode and the pipeline still
completes with clearly labelled placeholder visuals and espeak narration.

Docs: [ARCHITECTURE](docs/ARCHITECTURE.md), [DECISIONS](docs/DECISIONS.md),
[PROGRESS](docs/PROGRESS.md), [BLOCKERS](docs/BLOCKERS.md).
