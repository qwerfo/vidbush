# Vidbush build specification (reference copy)

This is the specification Vidbush was built against. It is kept in the repo
so any engineer or model can audit the code against it.

## 1. Product definition

Inputs:
- A, Brief: topic, target length in minutes, format (documentary or top10), audience notes, optional angle. Vidbush researches, outlines, writes the script, then produces the video.
- B, Script: any length, no word cap.
- C, Voiceover: an mp3 or wav narration, with or without the script. Transcribe and align if no script is given.

Outputs in output/<project_id>/:
- final.mp4: 1920x1080, 30 fps, H.264 high profile, AAC 192 kbps, -14 LUFS integrated, true peak -1 dBTP.
- captions.srt and captions.ass; chapters.txt (YouTube format from scene timestamps).
- description.md: 3 title options, description, tags, chapter list, source and footage attribution block.
- thumbnails/: 3 options, 1280x720 PNG.
- shotlist.json (the edit decision list), qa_report.json and qa_report.md, cost_report.json (every provider call with a cost estimate).
- sources/: every downloaded asset with a sidecar .json (source URL, licence, attribution text).

Capabilities: research agent with citations and a fact check pass; scriptwriter for documentary (acts) and top10 (countdown); shot list with a visual intent and type per beat; footage sourcing from YouTube (yt-dlp), Wikimedia Commons, Internet Archive, Library of Congress, NASA, Pexels and Pixabay when keys exist, with licence metadata and a domain blocklist; a vision judge that scores candidates and rejects burned-in captions, big watermarks, channel logos, talking heads, wrong subjects; AI images for uncovered shots with Ken Burns or DepthFlow; motion graphics (animated maps with routes and arrows, bar and line charts from sourced figures, number cards, title cards, lower thirds, callouts, highlight boxes, overlays); ElevenLabs narration with word timestamps and faster-whisper alignment fallback; captions of one line, 3 to 5 words, one consistent height in the bottom band; music bed from a local folder with ducking, optional sparse SFX; FFmpeg assembly with a segment cache; style profiles per brand plus a reference learner; a shot editor UI (preview, swap, upload, regenerate AI, re-render one segment); chat revisions mapped to structured edit operations; a thumbnail generator; brand profiles (voice, caption style, music folder, visual mix, intro and outro, blocklist, niche notes, language).

## 2. Constraints and principles

- No em dash character anywhere (code, docs, UI, prompts, generated text).
- Every model choice is an operator setting. Code asks for a seat by role; config maps roles to ordered providers; first that works is used.
- Fallback moves only the failed request. Never mix image providers within one production unless the primary is down for the rest of the run.
- A failing model is presumed to fail on our request format: adjust per model before writing it off. Never cap max_tokens by default; ask for concise output (an operator may set an optional per provider cap in seats.yaml).
- Probe a down provider every 10 minutes and return to it.
- Cost ceilings are config: image ceiling 5 cents, warn above 3 cents. Track every call.
- Visual bar is "good enough": fail only differences that break immersion. Judge place first, then subject, then action.
- Opening 30 to 90 seconds visually dense with real activity. No near-identical consecutive shots, no repeated framing twice in a row; derive full, tight and detail framings from one strong still.
- One Start per project; the pipeline runs to the end without input; approvals happen in the editor. Every stage writes to disk and can resume after a crash.
- Licensing: record source URL, author, licence, attribution for every asset; description attribution block generated from them; domain blocklist; allow_youtube flag (default true); flag_nc_nd flags CC NC or ND assets in qa_report; nothing blocked by default except the blocklist.

## 3. Stack

Python 3.11+; FastAPI with an in-process worker pool over a SQLite job table (no Redis, no Celery); SQLite via SQLModel at data/vidbush.db; FFmpeg and ffprobe; yt-dlp; PySceneDetect; Pillow and numpy; faster-whisper. Motion graphics: Remotion in renderer/ consuming a JSON motion spec, or an identical-interface Python frame renderer if Chromium is unavailable. Maps offline from Natural Earth, never a tile server. Charts rendered by the motion renderer, never matplotlib. UI: Vite and React in ui/, plain CSS, dark theme, served by FastAPI on port 8080. Config: config/seats.yaml, config/style_profiles/*.yaml, config/brands/*.yaml, .env with a documented .env.example. Packaging: pyproject.toml, python -m vidbush, Dockerfile, docker-compose.yml, Windows instructions.

## 4. Data model

Project, Script (Scene: id, title, purpose hook|setup|act|item|turn|climax|resolution|outro, beats; Beat: id, text of 1 to 3 sentences, facts_used, emphasis), Narration (audio, duration, words with start, end, beat_id), ShotList (Shot: id, beat_id, start, end, duration, type footage|footage_overlay|image|image_overlay|ai_image|ai_video|motion_graphic, intent, must_show, must_avoid, queries, overlay MotionSpec, candidates, selected, judge_score, judge_notes, render_segment_path, render_hash), Candidate (id, source, url, local_path, in and out points, licence, attribution, keyframes, score, reject_reason), MotionSpec (kind map|bar_chart|line_chart|number_card|title_card|lower_third|callout|highlight_box|text_overlay|arrow, duration, width, height, background transparent|solid|footage_behind, style tokens, payload, source_note), StyleProfile (visual_mix with full_screen_cards and animated_overlays, pacing per type with hook multiplier and card spacing, caption style, letterbox 2.39, transitions, colour LUT, music ducking depth, attack, release, bed gain, Ken Burns intensity, DepthFlow), Brand, EditOp (swap_shot, upload_shot, regenerate_ai, set_duration, split_shot, merge_shots, change_type, set_overlay, remove_overlay, reorder, retime_section, change_music, set_caption_style, rewrite_beat).

Default documentary mix (share of shot count), from a reference analysis of a top geography documentary that measured 20 raw footage, 21 footage with overlay, 7 raw image, 10 image with overlay and 42 animation. Animation in reference channels is mostly animated maps and overlays that move over footage and stills, with full screen cards used sparingly, so the profile splits animation into two numbers: full_screen_cards 12 (title, number, chart or map as a full frame; motion_graphic shots) and animated_overlays 30 (maps, charts, callouts, arrows and lower thirds composited over footage or stills). Animated overlays are counted on top of the footage or image shot they decorate: they are added to footage_overlay and image_overlay in the ratio of footage to still shots in the picture mix (41 to 17). Resulting shot type targets: footage 20, footage_overlay about 42.2, image 7, image_overlay about 18.8, motion_graphic 12, ai_image 0 (fill-in only), ai_video 0 (opt in). A map or chart used as an overlay renders as an inset panel over the picture. Profiles that still use the old motion_graphic key read it as full_screen_cards. Card spacing: at most one full screen card per 45 seconds outside the hook and the recap (profile setting pacing.card_spacing_s; structural cards such as top10 number cards always stay and count against free cards near them). Pacing: footage 3 to 7 s, images 4 to 8 s, motion graphics 6 to 15 s, 0.7 multiplier in the first 60 s, bounded randomisation. Letterbox on, captions on. Top10 adds a number card at each item start, an item title lower third, a recap grid at the end; its profile uses full_screen_cards 20 and animated_overlays 22 because the number cards are full screen by design.

## 5. Stages

1 Intake. 2 Research (brief only): 25 to 60 fact cards with sources, fact check marks verified, unverified or contradicted and drops contradicted; 5 to 10 visual anchors. 3 Outline at about 150 wpm. 4 Script (writer, parser for pasted scripts, transcription for voiceover). 5 Narration: ElevenLabs with timestamps, newest model by listing, chunk at scene boundaries, 40 ms crossfades, alignment check with faster-whisper, -16 LUFS. 6 Shot list: every beat covered, no gaps or overlaps, mix within 5 percent (shot type targets from the full screen card and animated overlay split), at most one full screen card per 45 s outside the hook and recap, pacing distributions, hook density, MotionSpecs grounded in fact cards and real coordinates. 7 Sourcing: parallel queries, YouTube section downloads, scene detection, 3 to 12 s clips, 3 keyframes, cap 8 candidates, attribution. 8 Judge: 0 to 100 scores, hard rejects, threshold 70, director loop (requery, re-source, convert to AI image), up to 3 attempts, never empty. 9 Visuals: AI images at 16:9 at least 1920 wide, Ken Burns or DepthFlow, derived crops, optional ai_video. 10 Motion graphics with 3 palettes and one open font, source notes on data graphics. 11 Captions and audio mix: 3 to 5 word lines at the band height, track choice by mood tags, sidechain ducking with one constant level, fades at scene boundaries, -14 LUFS. 12 Assembly with per-shot segment cache, letterbox, captions, intro and outro, mux. 13 Packaging: thumbnails, chapters, description, qa report, cost report.

## 6. Providers

OpenAI-compatible HTTP for LLM, vision and image (relays configurable without code); dedicated adapters for ElevenLabs, fal.ai, Runware, Pexels, Pixabay. Mock mode when a role has no working provider: canned LLM fixtures, deterministic vision scores from image statistics, labelled gradient placeholder images, espeak or tone-burst TTS; every mock call marked. Retry: 3 attempts with backoff on transport errors, one attempt then fallback on refusal or 4xx, 10 minute probe on a down provider.

## 7. Style engine

learn-style <url or path> <name>: scene detect, 2 keyframes per shot, classify with the cheap_classifier seat, duration distributions, letterbox and caption detection, write a profile with provenance. Mock mode: durations and band analysis only, mix unknown.

## 8. Quality

Post-assembly judge pass (one frame per 3 s) against intents, captions checked against narration words. qa_report lists every shot with score, reason, attempts, flags (mock, nc_nd, anachronism, low_res, no_match_accepted) and summarises percentage above threshold, flagged shots with timestamps, mix achieved versus target, animation share (full screen cards, shots with an overlay, map and chart insets) versus target, pacing stats, caption coverage, attribution count. Operator target: at most 1 percent no_match_accepted.

## 9. UI

Projects (list, create from brief, script paste, voiceover upload, brand, format, minutes). Project (stage progress, live log, player, shot strip, shot panel with intent, candidates, Use this one, Upload my own, Regenerate as AI image, Edit duration, Remove overlay; changes re-render one segment and re-assemble). Chat panel mapping messages to EditOps. Brands page. Settings page with seat status. Everything is also a CLI subcommand.

## 10. Acceptance criteria

A1 demo from fixtures/demo_brief.md: final.mp4 of at least 3 minutes with burned captions, narration, at least 20 shots covering at least five shot types, at least one map, one chart, one title card, ducked music, chapters, description with attribution, three thumbnails, qa and cost reports, in mock mode with no keys. A2 script mode and A3 voiceover mode pass the same checks. A4 serve on 8080: create, watch, swap one shot, see the re-rendered video. A5 revise "make the first 30 seconds faster and replace shot 7 with an AI image" produces and applies the expected EditOps. A6 learn-style on a local fixture writes a profile. A7 pytest passes including a mock demo integration test, run twice to prove the cache re-renders zero segments. A8 kill in stage 8 under an env flag, resume completes without redoing stages 1 to 7. A9 docker compose up builds and serves on 8080. A10 zero em dash characters.

## 13. Coding standards

Type hints, pydantic contracts, functions of about 60 lines at most, stages as pure functions from disk to disk, structured JSON logs to data/logs/<project_id>.jsonl and stdout with no secrets, one client wrapping every provider call, config loaded once and passed, tests offline with fixtures and mocks, pinned requirements, short contract docstrings.
