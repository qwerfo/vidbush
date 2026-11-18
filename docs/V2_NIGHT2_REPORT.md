# V2 night 2 report

Written 2026-11-18 about 21:15 (+0700). It replaces the 20:05 draft. Every number below was read again from the run files, logs and cost reports at 21:08 to 21:12.

## 1. Summary

- v2/base now joins all six V2 lanes: audit, sources, layouts, planner (BASE READY be327be), then picker (a59bd9e) and sound (25866ed).
- The clause planner runs end to end behind the switch `planner: clause`. The slot planner is still the default.
- Three projects ran on the clause path: alaska_v2a, alaska_v2b, dunkirk_v2.
- alaska_v2a is not rendered. Its 5 USD cost cap stopped it in the pick stage, on the interim edit path, after scene 1 of 11. alaska_v2b reused its pools.
- alaska_v2b is rendered (final.mp4, 20:45). It is not audited: the night's 15 USD limit was reached first.
- dunkirk_v2 is rendered (final.mp4, 20:25) and audited. That render was made before the fill v2 and pictureless layout fixes. A frame check of it found about 64 s broken: 23 s of empty layouts and 41 s where the fill picked film slates, leaders, burned subtitles or unrelated maps.
- dunkirk_v2 audit: real 42.7, AI 25.0, designed 30.9, on words 48.2, noticeable 38.2, overlay noticeable 62.2. It is worse than dunkirk_base on real share, on words and noticeable. Designed rose from 6.8 to 30.9.
- dunkirk_v2 is being rendered again from 21:09 with the fixes applied to its shot list. That render is not in this report.
- Eleven fixes went in during the renders (section 4). Each one came from something a render showed.
- Night cost: 15.54 USD. The largest avoidable part is the sourcing classifier on openlux: 4.61 USD at 0.01 USD a call.
- YouTube asked to sign in on every project. The paid download route took over. It cost 0.05 USD in all, but it rejected 42 videos below 720p after paying for them.

## 2. Merge

Order: audit, sources, layouts, planner into v2/base (BASE READY be327be). Then picker and sound, merged as soon as both said READY (fb999fd, 6546866). The second merge was done in a second worktree, so the running alaska_v2a kept its code.

What conflicted:

- Docs, everywhere. HANDOFF.md, DECISIONS.md and ARCHITECTURE.md conflicted in every merge. Every doc conflict kept both sides.
- HANDOFF renumbering. Two lanes both wrote a section 12. They were renumbered, not merged, so each lane's text stays as written: 12 measurement and replay, 13 sources, 14 layouts, 15 integration (new tonight), 16 picker, 17 sound. PROVIDER_NOTES only points at section 10f, which did not move. HANDOFF still has two sections numbered 11 (art layer, run 3 AI stills); that is older than tonight.
- seats.py imports. Both lanes' imports kept: the audit lane's reply cache and CostCapReached, the art lane's video_elevate. Nothing else in code conflicted there.
- models.py, the Shot.layout contract. The layouts lane owns the contract. Its LayoutKind, LayoutSpec, LayoutElement, LAYOUT_ROLES, StillSequence and the Shot fields layout_notes, layout_at and grade were kept. The planner's own LayoutSpec was dropped. Only its StyleProfile.planner flag was added back.
- After the merge the planner emits exactly the layouts contract. An element role the kind does not read takes the kind's list role or the next free role, because the renderer silently skips a role it does not know.
- Full suite at BASE READY: 1728 passed, 17 skipped. After sound: 1758 passed, 17 skipped.

## 3. How the pipeline is wired

The switch: `planner: clause` in the project's style profile (work/profile_override.json `{"planner": "clause"}`). The slot planner stays the default until the V2 audits beat the baselines.

The stage names do not change, so snapshots, resume and --force work as before:

1. intake, script, narration: unchanged.
2. shotlist: the clause planner. It splits narration into clauses, runs the director once per scene, and writes work/clauses.json and work/direct.json (the director's scene plans). Layouts are emitted in the layouts lane's contract.
3. sourcing: one source pool per scene, three scenes at a time. Shots carry the director's per-class queries and the real thing. Full layouts that show real pictures (name card, split comparison, list slideshow, document focus) source stills.
4. judge: for a clause project this is the picker lane's pick stage (judge/pick_stage.py). Each pool asset is described once (judge/describe.py, cached by content hash). The picker picks per scene by comparison. Code checks the pick (AI rulings, repeats, time ranges, identity, document words, labels). A clause with no pick goes down the ladder: re-source (off by default since 3c513f3), restate as a layout, hold the previous real picture, AI still where the rulings allow. After the picker, no shot may leave empty (18962d9, bc4ace5). AI stills are generated here, three shots at a time (6e04962), and checked with fixed questions. The AI rulings are measured again on the picked list (plan_v2.json ai_rulings).
5. visuals, motion: motion renders layouts and panels. If its worker pool breaks it renders the rest one at a time (7b0d3a5).
6. mix: sound v2 (music planned per scene, quiets, ambience, paper effects).
7. assembly, packaging: packaging runs the tile audit on the finished video and puts its six numbers at the top of qa_report.md. An audit failure never fails packaging.

YouTube: the free yt-dlp route is tried first. The first bot check switches the rest of the run to video-download-api.com (host p.savenow.to): 0.0002 USD per 720p MP4 job, 0.0001 USD per VTT subtitle job. The whole file is fetched once and cut locally. The route is capped per run by sources.youtube_api_budget_usd (0.50 USD).

## 4. Fixes during the renders

One line each, with the principle from the commit.

- 4a69db9 Review fixes. Principle: Adam's AI rulings count what is on screen, so a hold of an AI still never resets the AI run count; a document's stand in is never invented; a hold never outlasts its picture's limit or its clip.
- 1267c5c One pick path. Principle: the picker's describe, pick and ladder replace the main lane's interim edit and ladder, because two pick paths would drift.
- aff2227 Ladder re-source fetches at most 3 assets a clause. Principle: a repair step costs less than the sourcing it repairs. One Alaska scene took about 25 minutes at the 8 asset cap.
- 3c513f3 Ladder re-source off by default (judge.ladder_resource). Principle: a step with no wall clock bound does not run unattended. One scene's re-source ran over 40 minutes.
- 6e04962 AI stills generate three shots at a time. Principle: independent work waits only on its own provider limit. Serial takes were 244 s each, about 5 hours for alaska_v2b's AI shots.
- 18962d9 Empty shot fill, version 1. Principle: a plain slate is never a picture. A shot with no picture holds the previous real picture, else takes the best unused real asset from its pool with its words over it. On alaska_v2b 32 shots (about 2 minutes) were empty.
- be6f351 The pick stage reads the director's plans from direct.json, never from its own output. Principle: a stage reads only inputs it does not write, so a rerun sees what the first run saw. A rerun had paid again for every stored reply (about 0.55 USD on alaska_v2b).
- 9fcdfeb A rerun pick keeps AI stills it already made when they pass their check again. Principle: a rerun never pays to make a picture it already made. Each resume had regenerated every still (about 4 minutes and 0.01 USD a take).
- 763c866 An empty route map is never shown; the shot keeps its own picture. Principle: a graphic with nothing on it is worse than none. 2 alaska_v2b shots drew a blank map.
- cba3af9 A document whose page was not found becomes its words over related real footage. Principle: Adam's ruling for a missing document, applied after the pick. One empty document ground ran about 14 s on alaska_v2b at 0:38.
- bc4ace5 Empty shot fill, version 2, and pictureless layouts dropped. Principles: words go only over a picture with no words of its own, and a picture is chosen by what it shows, not by its search score. The fill now takes only described, clean, wordless pictures (no blocking quality, no presenter or screen, no document look, no text in frame), ranked by words shared with the narration and the shot's intent. A list slideshow, name card, split comparison or document with none of its real pictures is dropped and filled the same way. On dunkirk_v2 the old fill put about 41 s of leaders, slates, burned subtitles and unrelated maps under our text, and pictureless layouts drew about 23 s of empty boxes or black.
- 7b0d3a5 Motion renders the rest one at a time when its worker pool breaks. Principle: a slower stage beats a failed one. A dead worker failed motion three times on alaska_v2b; every clip renders fine serially. It fired again on the dunkirk_v2 re-render at 21:09.
- fe80a49 The fill warns when its describe lookup finds nothing. Principle: a lookup that finds nothing must say so. With the wrong describe folder every lookup missed and dozens of shots stayed empty with no warning. The folder can now be set with DESCRIBE_CACHE_FOLDER or describe_cache_folder.

Note: alaska_v2b ran once on commit c3db207 ("Ladder re-source fetches at most 3 assets a clause"). That commit is on no branch now; aff2227 replaced it with the same subject.

How the fixes reached the renders: alaska_v2b's pick ran last at 18:38 to 20:00. The post pick fixes (763c866, cba3af9, bc4ace5) were applied to its shot list at 20:41 as an operator edit, then visuals, motion and assembly ran again. So the alaska_v2b final.mp4 has fill v2. The dunkirk_v2 final.mp4 (20:25) has none of them.

## 5. The audit's six numbers

Audit numbers are tile audit shares in percent (every 2 s frame judged against its words). Cuts per minute (video) are PySceneDetect ContentDetector cuts on the finished video (threshold 27, min 15 frames, scratch/night2/cuts.py). Cuts per minute (shot list) are shots minus one over minutes.

| Project | real | AI | designed | on words | noticeable | overlay noticeable | cuts/min (video) | cuts/min (shot list) |
|---|---|---|---|---|---|---|---|---|
| Vidrush (vidrush-alaska) | 56.4 | 2.6 | 40.7 | 71.3 | 18.6 | 24.8 | 16.3 | n/a |
| alaska_run3 | 67.1 | 26.0 | 6.9 | 58.6 | 22.1 | 43.3 | 11.5 | 11.3 |
| alaska_v2a | not rendered | | | | | | n/a | 16.2 (planned, 180 shots) |
| alaska_v2b | not audited | | | | | | 14.7 | 15.3 (170 shots) |
| dunkirk_base | 65.9 | 26.8 | 6.8 | 55.0 | 27.3 | 28.1 | 10.4 | 11.6 |
| dunkirk_v2 (20:25 render, before fill v2) | 42.7 | 25.0 | 30.9 | 48.2 | 38.2 | 62.2 | 13.1 | 15.7 (116 shots) |

- alaska_v2a: stopped by its cost cap in the pick stage on the interim edit path. Not rendered.
- alaska_v2b: final.mp4 exists. Not audited, because the night's 15 USD limit was reached.
- dunkirk_v2: this audit measured the render made before the fill v2 and pictureless layout fixes. A frame check of that render found about 64 s broken: 23 s of empty layouts and 41 s where the fill picked slates, leaders, burned subtitles and unrelated maps. That is about 15 percent of its 440 s, and it falls in "on words" and "noticeable". The numbers are not a measure of the fixed pipeline.
- dunkirk_v2 was audited by openlux/claude-sonnet-5-5. The other three audits used closerouter/anthropic/claude-sonnet-5.5. Closerouter refused some Dunkirk tiles (it read the narration as the 2017 film). The auditor change may move the numbers a little.
- Tile counts: Vidrush 349 tiles (141 overlay), run 3 331 (60), dunkirk_base 220 (32), dunkirk_v2 220 (111).
- dunkirk_v2 overlay noticeable is 62.2 on 111 overlay tiles. The old fill's text over slates and subtitles is a large part of it.
- The packaging qa_report.md of dunkirk_v2 (20:33) has no audit numbers. Packaging ran the audit from 21:01 to 21:06, then its 4.95 USD cap stopped it at 21:07 before it wrote the report again. Its own frame check says 63 of 116 shots (54.3 percent) fail.
- Cuts: the V2 videos cut 13 to 15 times a minute, against 16.3 for Vidrush and 10 to 12 for run 3 and dunkirk_base. The shot lists plan 15 to 16.

## 6. Screen time by kind

From the tile audit (seconds and percent of tiles):

| Project | real footage | real still | AI | designed | other |
|---|---|---|---|---|---|
| Vidrush | 246 s (35.2) | 148 s (21.2) | 18 s (2.6) | 284 s (40.7) | 2 s (0.3) |
| alaska_run3 | 354 s (53.5) | 90 s (13.6) | 172 s (26.0) | 46 s (6.9) | 0 |
| dunkirk_base | 182 s (41.4) | 108 s (24.5) | 118 s (26.8) | 30 s (6.8) | 2 s (0.5) |
| dunkirk_v2 (20:25 render) | 120 s (27.3) | 68 s (15.5) | 110 s (25.0) | 136 s (30.9) | 6 s (1.4) |
| alaska_v2b | not audited | | | | |

Tile kinds: Vidrush 53 map and 29 document tiles; run 3 5 maps and 11 documents; dunkirk_base 4 maps and 7 documents; dunkirk_v2 16 maps, 2 documents, 49 designed cards, 55 AI images.

dunkirk_v2 against dunkirk_base: designed rose by 106 s, real footage fell by 62 s, real stills fell by 40 s, AI stayed about the same (110 s against 118 s). So V2 replaced real pictures with designed ones, not AI with designed ones.

From the shot list, by what each shot selects (scratch/night2_metrics.py). A shot whose selected clip is its rendered layout counts as designed. Other is a shot with nothing selected.

| Project | real footage | real still | AI | designed | other |
|---|---|---|---|---|---|
| alaska_run3 (final shot list) | 300.3 s (45.3) | 208.1 s (31.4) | 154.1 s (23.3) | 0 | 0 |
| dunkirk_base (final shot list) | 201.2 s (45.8) | 115.6 s (26.3) | 122.8 s (27.9) | 0 | 0 |
| alaska_v2a (planned, before pick) | 400.4 s (60.4) | 138.9 s (21.0) | 0 | 123.2 s (18.6) | 0 |
| alaska_v2b (rendered shot list, fill v2) | 289.5 s (43.7) | 78.4 s (11.8) | 88.1 s (13.3) | 183.3 s (27.7) | 23.1 s (3.5) |
| dunkirk_v2 (20:25 render) | 156.0 s (35.5) | 116.7 s (26.6) | 82.8 s (18.8) | 67.1 s (15.3) | 17.0 s (3.9) |

- AI shots: alaska_v2b 33 shots (88.1 s), dunkirk_v2 33 shots (91.7 s).
- On paper alaska_v2b moves toward Vidrush: AI falls from 154 s (run 3) to 88 s, designed rises from 0 to 183 s. On dunkirk_v2 the shot list said 83 s of AI, but the audit saw 110 s of AI tiles. Expect a gap like that on alaska_v2b too.
- Fill v2 on alaska_v2b: 40 shots (182.6 s) got a related real picture with their words over it, 3 shots (9.0 s) hold the previous picture, 3 shots (14.3 s) were left empty. So about 28 percent of the video is fill. Its quality is not measured.

## 7. Layouts, panels and documents: planned, reached screen, dropped

"Reached screen" means the rendered shot list selects the layout's clip or panel. It is not checked frame by frame.

| Project | layouts planned | after pick | in the rendered shot list | panels | documents shown | dropped, with reasons |
|---|---|---|---|---|---|---|
| alaska_run3 | 0 | 0 | 0 | 28 old overlay panels | 0 | none |
| dunkirk_base | 0 | 0 | 0 | 12 old overlay panels | 0 | none |
| alaska_v2a | 53 | not picked | not rendered | 6 planned | 4 planned | not rendered |
| alaska_v2b | 53 | 54 | 44, all reach screen | 2 (timeline) | 0 | 10 dropped after the pick for having none of their real pictures: 4 document focus, 3 split comparison, 2 list slideshow, 1 name card |
| dunkirk_v2 (20:25 render) | 17 | 18 | 18, 16 reach screen | 2 | 1 | route 'Sedan to Abbeville by road': an end did not resolve, no line; name card: no real portrait and the shot's picture is not real, no name shown |
| dunkirk_v2 (re-render from 21:09) | 17 | 18 | 14 | not measured | 0 | 4 dropped for having none of their real pictures: 2 list slideshow, 1 document focus, 1 name card |

alaska_v2b layout kinds in the rendered list: number over footage 13, route map 10, date title 7, split comparison 5, name card 4, timeline 2 full and 2 panel, list slideshow 1.

Route maps on alaska_v2b: 2 had nothing resolve ("an empty map"). Since 763c866 those shots keep their own picture; not checked on screen. Places that did not resolve: Santa Monica (route Santa Monica to Jacksonville, no line, 2 shots) and Hurricane. Why the 3 planned route map panels became full route maps after the pick (6 full planned, 10 after) is still not traced.

Documents: no V2 project found a real document page tonight. alaska_v2b shows 0 of its 4 documents. dunkirk_v2 showed 1 document panel in the 20:25 render; the re-render drops its document.

Other layout notes on alaska_v2b: 8 full layouts are held outside 3 to 7 s (1.8 s to 9.2 s); 1 full layout's first words come 1.8 s after its cut.

dunkirk_v2 re-render fill (applied 21:07): 27 shots (122.2 s) get a related real picture with their words, 6 shots (23.2 s) hold, 15 shots (54.3 s) are left empty because nothing real in the shot or scene passed the new rules.

## 8. Wall clock per stage

Seconds from the "stage done" log lines (last run). A stage stopped or killed has no done line; its time is from the log timestamps.

| Stage | alaska_run3 | dunkirk_base | alaska_v2a | alaska_v2b | dunkirk_v2 |
|---|---|---|---|---|---|
| intake | 8.1 | 0.0 | 0.0 | 0.0 | 0.0 |
| script | 0.1 | 0.2 | 0.1 | 0.3 | 0.2 |
| narration | 71.6 | 67.0 | 8.8 | 9.0 | 6.1 |
| shotlist | 638.4 | 330.3 | 1518.6 | from v2a | 579.7 |
| sourcing | 1895.1 | 2408.3 | 6919.0 | from v2a | 10913.1 |
| judge (pick) | 10257.2 | 7116.6 | 712, stopped at cap | 4898.9 last; 18181 over 6 runs | 981.3 last; about 7557 over 5 runs |
| visuals | 32.0 | 4.9 | not run | 4.0 last (4 runs) | 16.9 |
| motion | 400.6 | 93.4 | not run | 29.4 last (8 starts, 3 failed on a broken worker pool) | 172.5 |
| mix | 200.1 | 87.8 | not run | 155.0 | 108.6 |
| assembly | 2217.0 | 415.2 | not run | 81.4 last (406.9, 71.9, 81.4) | 318.0 |
| packaging | 1234.0 | 795.5 | not run | not run (night limit) | about 2498, stopped by its cost cap after the audit |
| total | 16954.1 | 11319.4 | about 9160 | 13616 with v2a's shotlist and sourcing | 13096 to assembly; about 15594 with packaging |

alaska_v2b pick runs, in order: 128 s (restart), 1758 s, 1869 s (on c3db207), 6266 s (killed for 6e04962), 3261 s (done), 4899 s (done again after 18962d9). Ladder re-source caused the long runs until it was turned off.

alaska_v2b after the pick: motion 194.6 s, mix 155.0 s, then the post pick edits at 20:29 and 20:41 reran visuals, motion and assembly. Motion failed three times on a broken worker pool (20:29, 20:31, 20:42). The last run (20:43) used the serial fallback and finished in 29.4 s.

dunkirk_v2 pick runs: about 35 s (closerouter describer 503, then stopped), 3886 s (cap 4.00), 2124 s (cap 4.40), 531 s (cap 4.75), 981 s (done under cap 4.95, 19:58 to 20:15).

Sourcing is the slow stage on the V2 path: 115 min for alaska_v2a and 182 min for dunkirk_v2, against 32 and 40 min on the old path. The paid YouTube route (whole file per video) and cold archive downloads are the likely cause. It is not split by source yet.

## 9. Cost

Per project, from cost_report.json (USD):

| Project | total | largest items |
|---|---|---|
| alaska_v2a | 5.23 | classifier on openlux 2.98 (298 calls); director on openlux 1.46; vision on openlux 0.77; video download API 0.0009 |
| alaska_v2b | 4.82 | picker on closerouter 1.36; describer on openlux 1.14; judge on openlux 0.91; editor 0.59; director 0.57; classifier on openlux 0.18; Algrow images 0.033 |
| dunkirk_v2 | 4.99 | classifier on openlux 1.45; judge on openlux 0.94; classifier on wellflow 0.59; describer on openlux 0.58; director 0.47; editor 0.41; picker 0.31; vision on openlux 0.07; video download API 0.049 |
| dunkirk_v2 audit | 0.50 | 241 auditor calls (audit/audit_cost_report.json) |
| Night total | 15.54 | the 15 USD night limit was reached, so alaska_v2b was not audited |

- alaska_v2b's cost report starts empty: the pools' cost stays with alaska_v2a.
- Baselines: alaska_run3 3.93 USD (1.84 of it ElevenLabs TTS), dunkirk_base 2.04 USD.
- The sourcing classifier booked 0.01 USD a call on openlux against 0.0001 on closerouter, because closerouter answered 503 for part of the night. That is 2.98 + 0.18 + 1.45 = 4.61 USD on openlux, plus 0.61 USD on wellflow (booked at an estimate). It is the largest avoidable cost.
- Earlier audits, from metrics.json: Vidrush 0.59, run 3 0.57, dunkirk_base 0.22. They are not in tonight's total.
- Sound lane proof: 1,890 ElevenLabs credits (0.34 USD), from the 25866ed commit.

Elevate image credits (subscription, 0 USD a call, 4 credits an image per config/seats.yaml):

- alaska_v2b: 110 images made (55 first takes, 55 retries), 4 rate limits (429), 3 refusals. 440 credits.
- dunkirk_v2: 84 images made (35 first takes, 49 retries), 2 rate limits, 1 refusal. 336 credits.
- Night: 776 credits, if failed calls cost nothing.
- Algrow fallback: alaska_v2b 3 images (1.05 credits, 0.033 USD); dunkirk_v2 1 refusal.

Firecrawl (web image search): queries from work/source_notes.json: alaska_v2a 71, alaska_v2b 10 (two ladder re-source runs), dunkirk_v2 28. Total 109. Credits are not recorded anywhere.

YouTube routes and the video download API (video-download-api.com, host p.savenow.to):

| Project | free route | paid route | rejected below 720p after paid download | API cost |
|---|---|---|---|---|
| alaska_v2a | 226 downloads, then a bot check about 94 min into sourcing | 3 MP4 jobs, 3 subtitle jobs | 5 Alaska videos over alaska_v2a and alaska_v2b | 0.0009 USD |
| alaska_v2b | 0 (bot check in each ladder re-source run) | 15 windows in ladder re-source, not booked | (counted above) | 0 booked |
| dunkirk_v2 | 0 (bot check 139 s into sourcing) | 174 MP4 jobs, 145 subtitle jobs | 37 | 0.0493 USD |

- Most rejected videos were 480p. The free route asks for 720p sections and never sees these files.
- Paid jobs are booked by the sourcing stage only, so alaska_v2b's re-source jobs in the pick stage are missing from its report. The missing cost is under 0.01 USD, but the report is wrong.

## 10. Other lanes

- Picker: READY at a59bd9e, merged into v2/base (fb999fd). Its live pick stage is wired as the judge stage of a clause project; it replaced the main lane's interim edit and ladder (1267c5c). Docs in HANDOFF section 16 and DECISIONS "V2: picker".
- Sound: READY at 25866ed, merged into v2/base (6546866). It ran in the mix stage of both renders. dunkirk_v2 ran cleanly: music 17.9 dB under speech (set 18), ambience 30.6 dB under, mix -14.0 LUFS, true peak -1.6 dBTP, no dead air, no sound faults. alaska_v2b also had no sound faults (37 effects, 6 skipped, true peak -1.5 dBTP). Its mix ran once at 20:04, before the 20:41 shot list edit, so a few paper effects may sit under layouts that were later dropped; not checked.

## 11. Format comparison

Files: E:\coding projects\Vidbush\compare\opening_letterbox.mp4 and opening_fullframe.mp4, the first 90.1 s of alaska_v2b (24 segments).

- Letterbox: as rendered, with captions. All 24 segments reused from the render, 8.1 s to cut.
- Full frame: full 16:9, captions off. All 24 segments and 6 layouts re-rendered, 67.1 s.
- The AI stills were prompted for letterbox, so full frame crops them more than they were made for. Judge the AI frames with that in mind.

## 12. Still broken or unproven, ranked by screen time damaged

1. Fill quality is not measured after v2. It fills about 183 s of alaska_v2b (40 shots) and 122 s of the dunkirk_v2 re-render (27 shots). Before v2 the fill was the largest measured fault: 41 s of slates, leaders, burned subtitles and unrelated maps on dunkirk_v2. The dunkirk_v2 re-render also leaves 15 shots (54 s) empty, alaska_v2b 3 shots (14 s).
2. alaska_v2b is not audited. All 662 s are unproven on screen.
3. AI stills fail their fixed checks 2 of 3 times, and 40 percent of AI shots retry. AI is 110 s (25.0 percent) of the dunkirk_v2 tiles, about the same as dunkirk_base and far from Vidrush's 2.6.
4. The paid YouTube route rejects many videos at 480p after paying for them: 37 Dunkirk, 5 Alaska. Real footage on dunkirk_v2 fell to 120 s of tiles, against 182 s on dunkirk_base.
5. Documents never found a page. 0 of 4 on alaska_v2b, and the one Dunkirk document is dropped in the re-render. They are now words over footage, by Adam's ruling, but the planned document shots are gone.
6. Route maps with places unresolved: 2 empty maps on alaska_v2b, Santa Monica and Hurricane unresolved, and Sedan to Abbeville on dunkirk_v2 with no line.
7. Pick stage cost and time. alaska_v2b took 18181 s over 6 runs; dunkirk_v2 took about 7557 s over 5 runs and hit its cap three times. On alaska_v2b the picker, describer and judge cost 3.41 of 4.82 USD.
8. Sourcing takes 115 to 182 min on the V2 path, against 32 to 40 min before. Not split by source yet.
9. The openlux classifier costs 0.01 USD a call, 100 times closerouter. 4.61 USD tonight.
10. A reused AI still is checked again with a vision call, because its brief is not stored. Small cost, but each resume pays it for every still.
11. dunkirk_v2 packaging was stopped by its cost cap after the audit, so its qa_report.md has no audit numbers. Booking gaps too: pick stage video download jobs are not booked, wellflow is booked at an estimate, Firecrawl credits are not recorded. No screen time at risk.

## 13. Adam's review list

- alaska_v2b, the one render with every fix: E:\coding projects\Vidbush\output\alaska_v2b\final.mp4. Watch the fill shots and the 3 empty shots (shot_015, shot_149, shot_161).
- dunkirk_v2, the audited render made before fill v2: E:\coding projects\Vidbush\output\dunkirk_v2\final.mp4 (20:25), audit in its audit\audit.md. A re-render with the fixes started at 21:09 and will replace this file.
- Format: E:\coding projects\Vidbush\compare\opening_letterbox.mp4 against opening_fullframe.mp4.
- Picker review sheets: work\pick_sheets in each project.
- Sound: work\sound_qa.md and work\sound_plan.json in each project.
- Rulings needed:
  - Is a clean related real picture with the words over it good enough for an empty shot, or should such shots go to AI or a designed card?
  - Letterbox with captions, or full frame with captions off?
  - Raise the night limit to audit alaska_v2b and the dunkirk_v2 re-render (about 0.5 USD each)?
  - Keep the paid YouTube route when it rejects 480p files after paying, or ask it for the best format first?
