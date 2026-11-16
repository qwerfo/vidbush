# Fable review of Vidbush (2026-11-16)

Scope chosen by Adam during the session: review and redesign only. No pipeline code was changed and nothing was
spent on live runs. This document is the diagnosis, the redesign, and the work split into lanes for later sessions.

Branch: `fable-review`, cut from local `live-run-3` at `0d2174c`. All evidence files are under
`E:\coding projects\Vidbush\fable_review\`.

## 1. Terms used in this document

- **Beat**: one unit of the script, one to three sentences. The Alaska script has 68 beats, median 9.6 s.
- **Clause**: a part of a sentence that names one thing a viewer could be shown. Usually 2 to 5 s of speech.
- **Shot**: one picture on screen between two cuts.
- **Slot**: a shot's time window and type (footage, still, card), fixed before any picture is chosen.
- **Idea**: one proposed picture for a beat, written by the director model or by regex code.
- **Candidate**: one found clip or still that could fill a shot.
- **Judge**: the vision model that gives each candidate one score from 0 to 100.
- **QA**: the check that runs on the finished video and writes `qa_report.md`.
- **Overlay**: text or a small map drawn over a picture (label, callout, inset map).
- **Designed material**: a full-frame composed layout: a comparison card, a number over darkened footage, a name
  card, a full-screen map, a chart, a text card.
- **Tile**: one frame sampled every 2 s from a finished video. The measure in this review is built on tiles.
- **Tile audit**: for each tile a reviewer records what is on screen, the words spoken at that moment, and:
  - **kind**: real footage, real still, document, AI image, map, chart, text card, designed card.
  - **match**: `on_words` (shows the thing being said now), `early` or `late` (shows what is said 3 s or more
    later or earlier), `related` (same topic, not the thing said), `generic` (fits any sentence), `wrong`.
  - **noticeable**: an ordinary viewer would notice a problem in this frame.
- **Scene**: a group of beats under one chapter title. The Alaska script has 11.
- **Anchor**: the exact narration words a picture or label belongs to.
- **Pool**: all candidates found for a stretch of narration, before any is chosen.
- **Treatment**: how a clause is shown: a plain picture, a named designed layout, or a map.
- **Seat**: a role (director, judge, image) mapped to a model in `config/seats.yaml`.
- **Floor**: the judge score (50) below which a candidate is not used while an alternative exists.
- **Requery**: a second search for a shot, with new queries, after the first found nothing usable.
- **Prefilter**: local checks that reject a candidate before the judge sees it (subtitles, logos, talking heads).
- **Mix**: the target share of each shot type in the style profile; the "mix draw" picks a slot's type from it.
- **`must_show`**: the list of things the director wants visible in a shot.
- **Lane**: one work stream of a session. "Run 3 lanes" are the earlier ones; lanes A to F are proposed here.
- **Vidrush**: the commercial tool. Adam's Vidrush render of the same Alaska script is the benchmark:
  `C:\projects\CaseFile\inbox\alaska\vidrush-alaska.mp4`.
- **Run 1, run 2, run 3**: the three finished Vidbush renders of the Alaska script, in
  `E:\coding projects\Vidbush\output\alaska`, `alaska_run2`, `alaska_run3`.

## 2. How this review was done

1. Seven independent readers mapped the code and stored run data without reading earlier reports
   (`fable_review\maps\*.md`). I read the shot planner and the judge myself.
2. Eleven independent reviewers audited every tile of four videos against the narration word timings
   (`fable_review\rows\*.jsonl`, 1,335 tiles; summaries in `rows\report_*.md`).
3. Only then one reader compared the earlier reports with the tile rows (`maps\earlier_vs_frames.md`).

Limits that apply to every number below:

- The tile reviewers are AI agents, not Adam. "Noticeable" is a judgement, and different reviewers covered
  different videos, so a gap of a few points between two videos is not proof.
- Cut counts come from ffmpeg scene detection at one threshold, with cuts under 0.3 s apart merged (flash
  transitions). Unmerged, Vidrush has 18.6 cuts per minute.
- Sampling is one frame per 2 s. Motion, audio and defects shorter than 2 s were not checked.
- Run 3 was finished by another session on 2026-11-16 at 09:23, while this review was running, from code
  at or near `0d2174c` (the exact commit that session ran is not recorded). It is the newest render and the fair picture of the current code.

## 3. The measured gap

| Measure (share of tiles) | Vidrush | Run 3 | Run 2 | Run 1 |
|---|---|---|---|---|
| Real footage, stills, documents | 55% | 72% | 27% | 8% |
| AI images, and video that looks AI-made | 2% | 26% | 70% | 37% |
| Designed material (cards, maps, charts, text) | 42% | 1% | 3% | 55% |
| Picture `on_words` | 64% | 52% | 35% | 46% |
| `early` or `late` | 8% | 16% | 32% | 27% |
| `wrong` | 2% | 5% | 16% | 14% |
| `related` or `generic` | 26% | 27% | 17% | 13% |
| Noticeable | 19% | 24% | 49% | 59% |
| Cuts per minute | 15.9 | 11.1 | 11.4 | 9.4 |
| Median shot length | 3.3 s | 5.2 s | 5.2 s | 6.0 s |

What this says:

- **Run 3 is a large improvement on run 2.** Noticeable tiles fell from 49% to 24% after the run 3 lanes
  (sourcing budgets, real over AI, picture anchors, floor). Different reviewers audited the two runs, so the
  size of the drop is approximate. That work should be kept.
- **Run 3 is now close to Vidrush on the count of visible faults, and still far from it in kind.** Vidrush is not
  mostly footage. It is 55% real material and 42% designed material, cut 40% faster, with almost no AI. Run 3 is
  72% real material, 26% AI, and has no generated designed material (its 1% is one found map clip).
- **Vidrush is beatable.** It has its own faults in 19% of tiles: research figures used as maps, a placeholder
  title "Portrait Image" on a name card (2:06), glitch transitions, the wrong highway highlighted (10:32).
- **The earlier documents name one of three gaps.** They say real footage is "the main thing that separates us
  from Vidrush" (`review_run2.json:876`). The tiles show three gaps: designed material, picture on the clause, and AI standing in for real things.

## 4. Root causes, ranked by screen time damaged

RC1 to RC4 are ranked by the share of tiles they damage. RC5 to RC7 touch every shot and cannot be ranked by
tiles; they are listed after.

Symptoms logged by earlier sessions are grouped under the cause that produces them.

### RC1. There is no designed material (about 40% of the benchmark's screen time)

- **Evidence.** Vidrush gives every comparison, number, list, name and route its own full-frame layout that
  builds on the words: split comparison "ALASKA: 733,391 PEOPLE / FORT WORTH" (1:20), "29 PEOPLE" over a darkened
  city (1:24), "REAL DISTANCE: 240 MILES = MAP DISTANCE: ABOUT 1 INCH" (3:36), a count-up "260 MILES" badge on the
  Seppala photo (5:50), an arc drawn Los Angeles to Cleveland during the sentence that names both (6:42). Only 7 of 65 Vidrush
  designed-card tiles are noticeable. Its found maps are weaker: 19 of 66 map tiles noticeable.
- **Run 3.** No generated full-screen card, map or chart. The script's central claims are never drawn. From 9:25 to
  10:02 the "map box" chapter shows a black frame, a tiny atlas page, two AI maps and an 1881 map with no Alaska.
  "Alaska is as wide as Interstate Ten is long" (9:58) has no line and no outline.
- **Cause.** Run 1 had cards, maps and charts on 55% of screen time, and its text cards failed (60 of 100 text-card tiles noticeable) because
  they were bare beige text restating the caption, held 6 to 12 s. The response was a stack of rules that removes
  cards: one card per 75 s, a card must be spoken at its start, a card must not repeat the narration, a card slot
  with no fresh card becomes footage. All 38 full-frame ideas in the run 3 plan (29 number cards, 8 maps, 1 chart) have the template
  wording of the regex code, not the director model's, and none reached the screen. The fault was the design of the cards, not their existence.
- **Symptoms this explains.** No charts, missing numbers, size comparisons never shown, small inset maps pasted
  over unrelated photos, callout boxes in the centre of the frame.

### RC2. Pictures for specific real things cannot be found, so AI or stock stands in (up to half of run 3)

Upper bound: 26% of run 3 tiles are AI and 27% are `related` or `generic`.

- **Evidence.** Vidrush shows the real thing a sentence names: the 1958 Anchorage Daily Times editorial "An
  Apology, and an Explanation" (0:46), the Billboard jukebox chart (0:32), the Texas Senate Journal amendment
  (1:04; a reviewer notes it may be a recreated page), Coldfoot Camp (3:30), the Balto unveiling photo (6:20).
- **Run 3.** The same sentences get the Texas Capitol at night (0:39), a satellite strip of lakes (0:49), and AI
  stills for much of the 1925 serum story (41 of 120 tiles between 4:00 and 8:00, plus 7 tiles of footage that
  looks computer-made) although real photographs exist. AI
  stands in for named real people and events: Lizzie (1:59), the Whalers' 2007 game (9:10, contradicting the real
  field photo at 8:37). Stock from Pexels is 26% of tiles and 38% of those are `related` or `generic`.
- **Causes, all in sourcing.**
  - No general web image search and no newspaper source. A web search key already exists for research
    (`config/seats.yaml:122`), unused for pictures.
  - Library of Congress returned 0 candidates in both runs: its bot-check cooldown (3,900 s) outlasts the stage.
  - Archive, Wikimedia, NASA, NARA and Library of Congress searches use the first three keywords and drop years; 67 to 97% of cached
    searches for the first three are empty (`maps\source.md` section 5, item 5).
  - The download cap counts clips, not assets: median 4 distinct assets per shot reach the judge.
  - Segment choice inside a long video uses speech, not pictures: transcript windows tie and the first wins;
    archive films are read from a fixed offset.
  - The director writes queries without knowing which sources exist or what they hold.
- **Symptoms this explains.** AI share, fabricated people and documents, generic stock at section joins, foreign
  stand-ins (a Japanese locomotive, a German pharmacy), three Texas Capitol shots in 15 s.

### RC3. Cuts are fixed before content exists (16% of run 3 tiles early or late, plus most overlay faults)

- **Evidence.** In run 2, 89 tiles were early against 18 late: pictures ran one sentence ahead for minutes at a
  time. Run 3 still shows it at beat joins and through the serum run (5:05 to 5:55): kettle before frozen hands,
  crash before the flip, Nome before the crawl. Vidrush mostly cuts at clause starts (8% of its tiles are early or late).
- **Cause.** `plan_slots` (`vidbush/shots/pacing.py:244`) sets every boundary, length and type from a duration
  distribution and a random draw. It sees word gaps and scene starts, but no picture idea and no beat boundary. The assembler then picks, for each
  fixed window, the idea whose anchor words are nearest. A cut never moves to its words. The director gives two
  ideas per beat whatever its length; beats get 0 to 4 shots, so extra shots take regex ideas or a neighbour's.
- **Measured on stored plans.** Shots whose anchor words fall inside their own window: 48 of 126 in run 2, 85 of
  126 in run 3. The run 3 anchor work helped; fixed windows cap how far it can go.
- **Symptoms this explains.** One-sentence lead, swapped pairs, slow pace (5.2 s median against 3.3 s), 28 of
  136 director pictures never shown, a still's query sent to a footage slot or the reverse (33 of 126 shots).

### RC4. Labels and maps are attached to words before the picture is known (about 10% of run 3 tiles)

- **Evidence.** 35 of 64 overlay tiles in run 3 are noticeable. "NOME, 1925" on modern Central Park (5:57).
  "About 7.6 meters tall" on a hose crew (6:53). A "Coldfoot to Deadhorse, 240 miles" map during "communities
  aren't served by roads", 22 s before its sentence (3:03). A "Texas and Alaska" map over a school bus (1:19).
- **Cause.** Overlays are planned with the idea, then moved between shots by timing rules. The picture under them
  is chosen later by the judge. Nothing checks the label against the chosen picture. A map counts as "spoken" when
  any drawn name is spoken, shaded regions included, and Alaska is shaded on most maps.
- **Rule to adopt.** A label appears only when the chosen picture is the real thing it names. Vidrush mostly
  works this way (Coldfoot Camp, Hurricane Turn) and is faulty where it does not (4:20, 9:22, 9:56).

### RC5. The judge is a noisy yes or no, used as a ranking (affects every shot; costs most of the wall clock)

- **Evidence.** In run 2, tiles on judge-passed shots were 48.6% noticeable, the same as the whole video. The
  thresholds at 50, 55 and 70 sit where almost no scores fall: of 1,305 run 3 candidates only 72 score between 40 and 69
  (973 below 40, 260 at 70 or more). Among shots with two passing candidates, the top two are within 3 points
  in 46 of 64, so list order picks the winner.
- **Cause.** One absolute score per candidate. The judge does not see the other candidates, the neighbouring
  shots, the video topic, the asset's description, or the frames that will be on screen (it sees 20, 50 and 80%
  of the clip; the render plays from the start).
- **Cost.** The judge stage is 171 minutes of run 3's roughly 4.7 stage hours.

### RC6. Format choices that mark the video as machine-made (all tiles, mild)

- Permanent 2.39:1 letterbox with a caption line in the bottom bar. The picture fills about three quarters of the
  frame height. Vidrush is full 16:9 with no running captions. This is Adam's call: see section 10.
- Narrow stills on wide blurred bars filling a tenth to a half of the width (0:49, 3:31 in run 3).
- One still motion for every still, two transitions, no colour grade across eight providers.
- Label boxes visible empty before their text (5:03, 6:47, 6:53).

### RC7. Nothing measures the output in a way that can be trusted or replayed (blocks all progress)

See section 6. This is ranked last by screen time and first by priority.

## 5. Architecture

The four questions from the brief, each answered with the evidence and a test.

### 5.1 Slots first, or footage first?

Slots first is the wrong order. The evidence is RC3: a boundary drawn from a distribution cannot land on a clause,
and a type drawn at random cannot know whether a real picture exists.

Proposed order, "clause first, pool aware":

1. **Segment** the narration into clauses from punctuation and word timings. Target 2 to 6 s. Cuts land on
   clause starts. This is deterministic code.
2. **Direct** each clause: one model call per scene, with the full script in context (1,841 words is small). For
   each clause the model states the subject the viewer should see, the treatment (picture, designed layout by
   name, or map), and the searchable real thing (entity, document, place, date).
3. **Source** into a pool per scene, not per shot. Each asset is described once by a vision model: what it
   shows, place, period, text in frame, quality, usable time ranges. Descriptions are cached by asset hash.
4. **Edit** per scene: one model call sees the clause list and the described pool together and assigns pictures
   to clauses by comparison. It may merge two clauses under one picture, and it must say "nothing fits".
5. **Fall back** by a fixed ladder when nothing fits: designed layout with real pictures inside, then map, then
   kinetic text over related footage, then an AI still. AI is never used for a named real person, a real event or
   a document.
6. **Label** after the pick, from the asset's own facts. No label on a stand-in.

Test before building all of it: run steps 1, 2 and 4 on the stored run 3 candidates (titles and keyframes stand
in for step 3's descriptions) for two scenes (the opening and
the serum run) and tile-audit the resulting shot list against section 3. No sourcing or render is needed.

### 5.2 Does the split between an idea model and a deterministic engine serve the output?

It fights it. The engine overrides what the model decides: the medium (a still idea in a footage slot for a 0.2
score cost), the count (two per beat), the framing (124 of 126 shots "full"), continuity (reuse is penalised
although the prompt asks for it), and whether a card plays. The shot planner holds 104 rules
(`maps\shots.md` section 2); the same intent is enforced in up to seven places, and several contradict each
other (card minimum 6 s against a static card cap of 4 s; a mix target of 68% overlay shots ending at 19%).

Proposed split: the model decides content and treatment per clause. Code enforces only what is physical or legal:
word timing, reading time, hold limits, legibility, licence, no repeated asset. That removes the mix draw, the
type-run rules, the card spacing rules and the idea scoring weights.

### 5.3 Is one judge score per candidate the right unit?

No (RC5). Replace it with three things: a description per asset (cached, reusable across shots and runs), a
comparative pick per clause with named checks, and frames that match what will play. The code already concedes
this in three places: the fixed questions for AI stills, QA's named checks, and a regex over the judge's reason.

### 5.4 Are the models given what they need when they decide?

No. `maps\prompts.md` has the full table. The main gaps:

| Model call | Decides | Is not given |
|---|---|---|
| Director | every picture and query | the script outside 8 beats (only a summary of 120 to 160 words), shot count and lengths, which sources exist, what was found |
| Judge | every pick | video topic, other candidates, neighbours, asset description, the frames that will play |
| Requery | new searches | the narration, the topic, what the failed searches returned |
| AI scene brief | every AI still | that the image model cannot draw text or maps; the frame shape |

## 6. Measurement

### 6.1 What is wrong now

- QA and the frames disagreed in every run. Run 2 QA reported 90.5% of shots at or above threshold; the tiles
  put 35% on the words. QA said run 2 improved on run 1 (72.6% to 90.5%); `on_words` fell from 46% to 35%.
- The headline QA number is the pre-render judge score. The post-render check is not a gate: `acceptance.py`
  only requires that `qa_report.json` exists.
- The earlier frame review sampled one frame per shot and found 19 shots off their words; the tiles show 48
  shots with an early or late tile (70 of 130 when wrong pictures are counted too). Its calibration set treated unviewed shots as passes, so QA's false alarm rate was overstated: 7 of the 10
  listed false alarms are noticeable in the tiles.
- Nothing can be replayed. `shotlist.json` is overwritten by five stages. No model reply is stored.
  `--force judge` does not re-score. `source/replay.py` looks up the wrong cache key for the three-keyword sources.

### 6.2 A measure Adam can trust without watching

The tile audit used in this review, made a command:

- `python -m vidbush audit <project>`: one frame per 2 s from `final.mp4`, one row per tile with the keys defined
  in section 1, written to `audit.jsonl`, and six numbers in `audit.md`: real share, AI share, designed share,
  `on_words`, noticeable, overlay tiles noticeable.
- **Targets**: `on_words` 64% or more and noticeable 19% or less (the benchmark's values); AI 5% or less
  (benchmark 2%); overlay tiles noticeable under 15% (stricter than the benchmark's 21%, because run 3's are 55%
  and labels are cheap to get right).
- **Trust comes from calibration, once.** Adam marks 40 tiles (20 flagged, 20 not) as "would notice" or "fine".
  The audit is trusted when it agrees on 34 or more. Two reviewers on one range must agree on noticeable for
  85% of tiles; that variance was not measured in this review.
- The tile audit judges the finished frame against the words, so it does not share the judge's blind spots.

### 6.3 A measure a session can iterate on in minutes

- **Stage snapshots**: each stage writes `work/<stage>.shotlist.json` and never overwrites another stage's file.
- **Reply cache**: every model call stores its reply under a hash of model, prompt and images. A replay with
  unchanged prompts costs nothing.
- **Plan replay**: rebuild the shot list from stored narration and pool and report, without rendering: share of
  shots whose anchor words are inside their window (run 2: 38%, run 3: 67%), cut rate, treatment mix, clauses
  with no picture. A prototype of the first number is `fable_review\scripts\anchor_offset.py`.

## 7. Every model prompt

Full verbatim text and a cold reading of each of the 22 prompts is in `fable_review\maps\prompts.md`. The ones
that shape the picture:

| Prompt | Asks | Withholds | Failure it invites |
|---|---|---|---|
| Director (`shots/seat.py:52`) | exactly 1 footage and 1 still idea per beat, with queries | full script, shot count, sources, found footage | too few ideas for long beats; unfindable queries; rule 10 says "never coordinates" while the schema example requires a map centre, which code then uses to drop real places |
| Judge (`judge/judge.py:40`) | one score, 70 means usable | topic, neighbours, other candidates, description | "good enough" and "contradiction is wrong_subject" in one prompt; `wrong_subject` is 549 of run 3 rejections |
| Requery (`judge/director_loop.py:55`) | broader queries | narration, topic, what failed searches returned | generic stock; reverses the director's "most specific first" |
| AI scene brief (`visuals/ai_images.py:416`) | a named outdoor place and period for every still | that text and maps cannot be drawn | a classroom wall map gets "Alaska, coastal mountain ranges, spruce taiga" appended |
| AI safety rewrite (`ai_images.py:92`) | "change wording, never content" and "remove names, text, maps with words" | nothing | the two orders collide; 200 lines of repair code follow |
| Rendered QA (`judge/qa.py:297`) | "be strict" | nothing important | stricter than the judge by design, so 41% of accepted shots fail and nobody can act on it |
| AI video (`visuals/ai_video.py:27`) | uses the still's title | the title is always "AI generated still" | dormant bug: the prompt has no scene content |

Patterns across all prompts:

- Examples from this one script are fixed into general prompts: dog team and sled (four prompts), "flat treeless
  Arctic town", and every director example is the Panama Canal.
- No prompt is told what its output is used for or what happens when it fails.
- No structured prompt gets its own validation error back; the retry resends the same text.

## 8. What is sound and should not be touched

- Narration, word timing and caption sync. All three runs pass the audio and video length check
  (`work/assembly.json`, `durations_match`).
- Map verification: offline geocoding, routes along real rail, road and river lines, distance labels checked
  against the drawn line. Keep it and use it for full-screen maps.
- The run 3 lanes: sourcing budgets and extra sources, real over AI at the same quality, picture anchors, the
  floor, "a judge outage is waited out". The drop from 49% to 24% noticeable followed them.
- The local prefilter (burned-in subtitles, corner logos, talking heads).
- Segment rendering, the segment cache, stream-copy concat, per-call cost entries.
- Seat policy in `config/seats.yaml`. No change proposed; `allow_mock_fallback` untouched.
- Number cards and callouts land on their words when they are allowed to play.

## 9. Other findings

- **Rights exposure.** 414 of 418 YouTube candidates in run 3 are Standard YouTube Licence, and YouTube is 21%
  of run 3 tiles. This is a monetisation risk for a channel, separate from quality. Adam's decision.
- **Cost.** A run costs 2.09 to 3.93 USD and 47 to 88% of that is narration audio. Money is not the limit;
  wall clock is (judge 171 min, sourcing 32 min in run 3).
- **Cost data is wrong in three places.** The primary image seat is recorded at 0 USD (76 of 79 image calls in run 3). The provider `openlux` charged
  about 37 times `closerouter`'s price per token for the same model name in run 3 packaging. The cost cap acts
  only between stages and `run` has no cap flag.
- **Run 3 history.** On 2026-11-15 name resolution failed during the judge stage; 301 judge calls were stored as
  score 0 and the stage reported done. The process later ended at the start of the motion stage with no error
  logged. Commit `0d2174c` addresses this. Another session resumed the run on 2026-11-16.
- **Dead or misleading code.** The Remotion renderer cannot run (no `node_modules`). `ai_video` and `depthflow`
  are off. An unparseable judge reply scores 50, which equals the floor, so it can ship.
  `test_reuse_still_costs_more_than_any_timing_gap` asserts the opposite of the constants.
- **Operator comments become prohibitions.** A review comment is written whole into `must_avoid`
  (`vidbush/review/fixes.py`), so "I want steam shovels" reaches the judge as a thing to avoid.
- **A year label cannot be timed.** Narration spells years out ("nineteen fifty-nine"); the matcher looks for
  "1959". None of the 12 regex year labels in run 3 could be placed.

## 10. Where this review disagrees with earlier decisions

| Decision | Evidence against |
|---|---|
| `DECISIONS.md:11`: a deterministic pacing engine assigns typed, timed slots, which "keeps the visual mix and pacing exact regardless of model quality" | RC3. The mix target was 0% AI and run 2 shipped 70%; cuts cannot land on clauses. |
| `DECISIONS.md:234` and `:235`: full-screen cards cut to 5% of the mix, card spacing 75 s | RC1. Vidrush is 42% designed material and its designed cards are rarely faulty (7 of 65 tiles). Run 1's cards failed on design and hold time. |
| Prompt rule 3: on-screen text never repeats the narration | Vidrush types key phrases and numbers as they are spoken. The rule exists because captions are always burned in, which is itself open (below). |
| Real over AI as the main goal of run 3 | Necessary, not sufficient. In run 2, 62% of real footage tiles were noticeable. Run 3 reached 72% real and still has no designed material. |
| Lower thirds allowed when the name is in `must_show` | `must_show` describes the wish, not the chosen picture. RC4. |

Open decisions that are Adam's, not a model's:

1. Letterbox and always-on captions, or full 16:9 with on-screen text only where it adds something, as Vidrush
   does.
2. Standard YouTube Licence footage: keep, restrict to Creative Commons, or flag for manual clearance.
3. AI stills for named real people and real events: this review recommends never.

## 11. Work for the next sessions

Lanes are ordered. Lane A comes first because every other lane is proved with it. Lanes B, C and D can run in
parallel after A: the treatment names B needs are the seven layout names listed in lane C, fixed here. Lane E
depends on B and D; until E exists, B's edit step reads stored candidates with their titles and keyframes. Suggested models are a starting point for the planner.

### Lane A. Measurement and replay (Opus)

- **Build.** The `audit` command, stage snapshots and the reply cache of section 6.2 and 6.3. Fix
  `source/replay.py` (`OfflineAdapter` must copy `keyword_words`; ignore TTL in replay). Make `--force judge`
  re-score. Add `--cost-cap` to `run` and check it inside the judge stage.
- **Delete.** The `pct_above_threshold` headline in QA; the fixed-grid frame sampler; `score_post_render`.
- **Proof.** `audit` on run 2, run 3 and the Vidrush file reproduces section 3 within 5 points on real share,
  `on_words` and noticeable. Adam's 40-tile check agrees on 34 or more. A second replay of the run 3 judge stage
  with unchanged prompts costs 0 USD.

### Lane B. Clause-first planning (Fable)

- **Build.** Steps 1, 2 and 4 of section 5.1 as a new planner beside the old one, selected by a profile flag.
  The director call receives the full script, the clause list with times, the source list and the treatment
  names from lane C. Output per clause: subject, treatment, real thing to find, queries per source class.
- **Delete when proved.** `shots/pacing.py` slot planning and the mix draw; the idea scoring weights in
  `shots/assembler.py`; in `shots/rules.py` the type-run, card-run, card-spacing and overlay-moving rules
  (about rules 2 to 8, 44 to 65 and 73 to 89 in `maps\shots.md`); the regex picture ideas in `shots/ideas.py`.
- **Proof.** Plan replay on the stored run 3 pool: anchor words inside the shot window for 90% of shots or more
  (now 67%); 14 to 17 cuts per minute; no clause without a treatment. Then one render: `early` plus `late` at
  8% or less.
- **Why Fable.** It replaces the core and must delete more rules than it adds.

### Lane C. Designed layouts (Opus for the build, Sonnet per template after the first two)

- **Build** in the Python renderer, each layout taking real pictures and building on word timings:
  1. Split comparison: two framed pictures with captions, left on its word, right on its word.
  2. Number over darkened footage, landing on the spoken number, with its unit.
  3. Name card: real portrait, name, role line. Only when a real portrait was found.
  4. List slideshow: one framed picture per spoken item.
  5. Date title over related footage.
  6. Full-screen route map: relief or satellite base, each place labelled as it is named, the line drawn between
     the two words. Uses the existing map verification.
  7. Equals card for unit comparisons ("240 miles = about 1 inch").
- **Rules.** Every layout holds 3 to 7 s, moves the whole time, and never shows a count-up value that is false
  at the moment it is read.
- **Delete.** Bare text cards and the beige title card; inset maps over unrelated pictures; centre callouts.
- **Proof.** Tile audit of a render: designed material 20 to 40% of tiles with under 15% of those noticeable
  (Vidrush: 42% and 21%).
  Vidrush frames to match are listed in `rows\report_5.md`, `report_7.md` and `report_2.md` under "Editing
  grammar".

### Lane D. Supply of specific real pictures (Sonnet per source, Opus for the pool)

- **Build.**
  - A web image search source using the existing search key, with page title and description kept.
  - A newspaper source (Chronicling America) and a working Library of Congress source.
  - Keep the year and the subject in archive queries; search Wikimedia by entity name first.
  - Cap downloads by distinct assets, not clips.
  - Store description, date, query and rank on every candidate.
  - Choose the segment inside a long video from sampled frames, not from transcript ties or a fixed offset.
- **Delete.** The title-word reorder of YouTube results; the title-only relevance gate; the fixed 10% and 20 s
  offsets.
- **Proof.** Replay on the Alaska script: a real candidate that names the thing exists for each of these 12
  clauses: the Anchorage Daily Times apology, "Texas No Longer Largest", the Texas Senate amendment, the jukebox
  chart, Coldfoot, the Hurricane Turn, Toksook Bay, the Bethel prematernal home, Seppala with Togo, Kaasen with
  Balto, the Balto statue unveiling, the Barrow football field. Vidrush found at least 8 of them.

### Lane E. Pool description and comparative edit (Fable to design, Opus to build)

- **Build.** Steps 3, 5 and 6 of section 5.1: describe each asset once, pick per clause by comparison, the
  fallback ladder, labels from asset facts.
- **Delete.** The 0 to 100 judge score and its thresholds (70, 55, 50), the reuse and short-clip penalties, the
  borrow-from-beat rule, the branding regex, the AI-still-as-default fallback, `label_ok`.
- **Proof.** Replay on run 3: tiles on picked shots are less noticeable than tiles on rejected alternatives
  (the current judge shows no difference). A render: AI 5% or less, overlay tiles noticeable under 15%, judge
  stage under 45 minutes.

### Lane F. Small fixes that stand alone (Sonnet)

Each is one small change with a test.

1. An unparseable judge reply is retried, never scored 50.
2. A requery model error falls back to heuristic queries instead of aborting the shot.
3. Spelled-out years match year labels.
4. A label box is never drawn without its text.
5. A still narrower than half the frame is not used full screen.
6. `ai_video` prompt uses the scene, not the title "AI generated still".
7. Image generation cost is recorded; a provider charging over 5 times another for one model is logged.
8. Remove this script's examples (dog team, Arctic town, Panama Canal) from general prompts.
9. Operator review comments are not written into `must_avoid`.

### What Adam should watch

Run 3 (`E:\coding projects\Vidbush\output\alaska_run3\final.mp4`) beside the Vidrush file.

| Video | Time | Judge this |
|---|---|---|
| Run 3 | 0:39 | Texas Capitol under "the editor of the Anchorage Daily Times printed an apology". |
| Vidrush | 0:46 | The same line: the real editorial on screen. Is this the standard you want? |
| Run 3 | 1:19 | A "Texas and Alaska" map panel over a school bus. |
| Vidrush | 1:20 to 1:32 | Split card "Alaska: 733,391 people / Fort Worth", then "29 PEOPLE". Do you want layouts like these? |
| Run 3 | 3:03 | A road map during "aren't served by roads", 22 s before its own sentence. |
| Run 3 | 5:05 to 5:55 | The serum run: does each picture arrive one sentence early? Do the AI stills bother you? |
| Run 3 | 5:57 | "NOME, 1925" on modern Central Park. |
| Vidrush | 5:50 | The "260 MILES" count-up on the real Seppala photo. |
| Run 3 | 9:25 to 10:02 | The map box chapter: is the shrunken Alaska ever shown? |
| Run 3 | 2:31 and 3:17 | Two stretches that work (ice road, Dalton Highway). Confirm this is "good enough". |

Then mark 40 tiles for lane A (section 6.2), and answer the three open decisions in section 10.
