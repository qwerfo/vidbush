# V2 night 3 report: fix and watch (branch v2/fix)

Date: 2026-11-19. Branch v2/fix from v2/base cdb6c79. Worktree E:\coding projects\Vidbush\worktrees\v2-fix.
Decisions: docs/DECISIONS.md, section "V2: fix". Fault lists of every pass:
E:\coding projects\Vidbush\fable_review\rows\night3\open_pass1_faults.md.

## Summary

- The rulings A to H, the sound gate and the cost rule are in the code, with tests. Full suite: 1924 passed, 17 skipped.
- The opening loop ran five render passes of the first 90 s of Alaska (alaska_v3_open). Each was checked as a
  viewer would see it: a frame every 0.5 s with the words spoken, read by an agent that had not seen the
  changes. Passes 2 and 4 got that blind read. Pass 5 got only the code checks, because its shot list already
  showed the regression.
- The loop did not reach two clean passes. The best opening is pass 4: 8 blank frames left (from 41), 2 high
  faults (from 4), sound in sync.
- The full renders alaska_v3 and dunkirk_v3 were not finished. The 12 USD cap was reached while both were in
  the pick stage (see Cost).
- The middle-stretch loop (step 3, second half) and the full-render checks (step 4) were therefore not done.
- The cause of the overspend is outside the code. closerouter refused every call all night for lack of balance
  (http 402), and openlux was down for long spells. The picker and director fell to anthropic_direct at about
  0.11 USD per picker call, about 7 times openlux for the same model.

## Step 1. Sound

What was measured on night 2's files:
- alaska_v2b and dunkirk_v2 final.mp4 both carry one AAC LC stereo 48 kHz track. It is the default track, its
  MP4 track header volume is 1.0, and its edit lists are the same as in run 3.
- The track sits at -17 dB mean in every 10 s window, from first to last, the same as mix.wav and
  final_audio.wav.
- The narration stem is in the mix at -19 dB.
- A transcription of the first 30 s of each final matches the script word for word, with timing within about
  0.3 s.
- The container is byte-for-byte the same in structure as run 3's, which Adam heard. The 90 s compare clips
  also carry audio.

Finding: the stored files are not silent. The silence Adam heard does not come from the render as stored, so
there was no render cause to fix. What changed instead is that a silent render can no longer ship:
- The assembly stage checks final.part.mp4 before it replaces final.mp4.
- In every 8 s window with speech, 90 percent of the narration's speech frames must have sound in the final.
  The speech-band envelopes must correlate at 0.6 or more, at a lag within two frames.
- Narration longer than the shot list also fails the stage.
- The numbers go to work/sound_check.json.
- Calibration:
  - alaska_v2b, alaska_run3 and dunkirk_v2 pass, at 0.995 or more correlation in every window.
  - Music only, silence, a 0.2 s shift and a cut file all fail.
- Every night 3 opening render passed it.

Worth checking on Adam's side: the player's volume or output device, or a muted browser tab. The stored files
play with speech.

## Step 2. Rulings: changed and deleted

A. On-screen text
- Changed:
  - The director is told that text appears only for a number with its unit, a name, a date, a short graphic
    label, or a passage marked on its page.
  - Validation rejects element text that copies four or more narration words in a row and has no number.
  - The picker's restate step may type only numbers, dates and labels.
- Deleted:
  - text_over_footage in emit and in the ladder's restate.
  - The fill's "first six words over a related picture".
  - The map fallback.
  - The document's quoted words over footage.
  - The renderer's quote-alone and quote-beside document modes.
  - The name lower third.
  - The centred text fallback of a failed motion spec.
  - The title_card placeholder on layout shots.

B. Fill order
- The order is: pick, then the picker's fitting real picture (plain), then a restated layout only when it is
  complete, then a neighbour held or started early up to 7 s, then an AI still, then the picker's loose real
  picture.
- Last resort, added tonight: a real photograph or film from the whole film's described pool whose recorded
  place and period agree with the clause's setting.
- The picker names its fitting and loose pictures and its hold answer in the same call as the pick.
- AI stills:
  - Never for a named real person or a document.
  - Real events are allowed again, as Adam ruled.
  - Never in the first 15 s.
  - At most two in a row, each at most 4 s.
  - The brief carries the setting, and the prompt rewrite must keep the season and year.

C. Fit
- The director gives every clause a setting: period, season, place, circumstance.
- The picker is told that fit is the situation, not a shared noun, and that a modern picture covers a
  historical line only when its subject is unchanged by time.
- Stand-ins (fitting, loose) are photographs or film of the real world: never a page, a drawing or a
  low-resolution picture.
- Feature film clips, unreadable signs and burned-in titles are refused.
- Deleted: the fill that ranked pictures by words shared with the narration.

D. Panels
- The director's default is a panel over a good picture, placed by the picture. A route map, chart, table or
  timeline is planned wherever places, sizes or quantities move, never a photograph of a map.
- Result: in the Alaska opening, the director's plan went from 0 panels and 6 full maps in its first scenes to
  a route map panel, a chart panel and two documents.
- Places are written 'Town, State or Country', and 'Town, ST' resolves the US state.

E. Whole graphics
- A layout or panel is shown only when every element, picture, place and route can be drawn and it has 3 s
  from its first word.
- A full layout whose words come late enters on its words over a real picture.
- A dropped layout leaves a real picture.
- A panel with no picture beneath goes full screen. A number or date with no footage is dropped and the shot is
  filled.
- A bar for a positive value is always visible.

F. Faces
- Faces are found once per file on the uncropped picture.
- Every crop keeps each face box whole: name card portrait, comparison sides, list items, still cover, still
  moves, derived crops. When no crop can, the picture goes on its blurred fill, and a name card is dropped.
- Not yet covered: video cover crops and explicit Ken Burns moves. Pass 4 shows the top of a head cut by a
  video crop at 22.5 s.

G. Repeats
- One asset serves at most two shots in a row and never appears again within 60 s. This is checked in the
  picker's code checks and enforced again on the final shot list.
- A hold is one shot with one move, not a new shot with a restarted move.

H. Opening
- The director is told the first 60 s are the strongest stretch.
- Code refuses an AI picture, still or layout element, in the first 15 s.
- The art layer is off for clause-planner projects. It added a third AI picture in a row after the picker.

Cost rule
- Describe, classify and audit run on the seat's cheapest working provider. One more than 5 times dearer is
  never used for them.
- When every cheap provider is down, the step stops with CheapSeatDown.
- Measured tonight: the describer, priced 0.0002 per call on closerouter, refused openlux (0.0035) and
  wellflow (0.02) all night. It used only stored records, so there was no describe spend.

## Step 3. The opening loop (first 90 s of Alaska)

| Pass | Blank frames (of 184) | High faults | All faults | What changed before the next pass |
|---|---|---|---|---|
| 1 | not rendered | 8 empty shots in the shot list | n/a | the picker must name a loose picture; short clips are slowed; state codes resolve; describe cache folder set |
| 2 | 41 | 4 | 13 | alternates follow the 60 s rule, not "never again"; a layout clause's stand-in may hold 12 s; a panel with no picture goes full screen; an empty shot is split between neighbours; feature film clips, unreadable signs and burned-in titles refused; AI prompt keeps season and year; small bars visible |
| 4 | 8 | 2 | 11 | stand-ins are photographs or film only; AI takes reused only for their own brief; a place label needs a confirmed picture; every open clause names a loose picture |
| 5 | 47 | 1 (23 s black) | code checks only | stricter stand-ins left a 23 s stretch with no usable picture in its pool; added the setting fill from the whole film's pool; art layer off |

Pass 3 was stopped before rendering when the pass 2 fault list arrived.

Pass 4 fault list:
- High:
  - 4 s black at 81 s (fixed for pass 5).
  - A 1942 front page held 9.5 s for a 1958 apology (fixed: pages are no longer stand-ins).
- Medium:
  - A place title over another town.
  - Two AI stills of the same story showing two different men, with snow against "summer".
  - A small static atlas map held 8 s at 10 s.
  - A clip-art cartoon held 9 s.
- Low: a tiny satellite card for 1.5 s, the size chart panel not placed, a tight video crop on a head, stacked
  route labels, two AI-looking Manhattan stills.

Pass 6, with the setting fill and the art layer off, could not run. Its first live call met the cost cap, and
the stored picker replies did not match the new requests (see What is still wrong).

The middle-stretch loop was not run.

## Step 4. Full renders

- alaska_v3: forked from alaska_v2a's pools; new director over all 11 scenes. The pick stopped at its cap
  (5.16 USD) inside the pick stage.
- dunkirk_v3: forked from dunkirk_v2's pools; new director over all 7 scenes. The pick stopped at its cap
  (3.33 USD; cap lowered to 3.3 to keep the night under 12) inside the pick stage.
- Neither was rendered, so there is no tile audit, sound check or frame check for them.
- To finish them once closerouter has balance (stored picks and descriptions are reused):
  `cd "E:\coding projects\Vidbush\worktrees\v2-fix" && PYTHONPATH=. python -m vidbush resume alaska_v3 --cost-cap 7`,
  and the same for dunkirk_v3.

## Step 5. Numbers

Audit numbers (tile audit, from night 2; nothing new could be audited tonight):

| Project | real | AI | designed | on words | noticeable | overlay noticeable | cuts/min |
|---|---|---|---|---|---|---|---|
| Vidrush | 56.4 | 2.6 | 40.7 | 71.3 | 18.6 | 24.8 | 16.3 |
| alaska_run3 | 67.1 | 26.0 | 6.9 | 58.6 | 22.1 | 43.3 | 11.5 |
| alaska_v3 | not rendered | | | | | | |
| dunkirk_base | 65.9 | 26.8 | 6.8 | 55.0 | 27.3 | 28.1 | 10.4 |
| dunkirk_v3 | not rendered | | | | | | |

Opening excerpt, from the rendered shot lists (92 to 97 s each):

| Pass | Shots | Cuts/min | Picked | Fitting or loose | Holding a neighbour | AI stills | Layouts | Empty | Longest AI run |
|---|---|---|---|---|---|---|---|---|---|
| 2 | 20 | 12.4 | 4 | 1 | 3 | 6 (17.5 s) | 5: route map panel, chart panel, chart full, date title, split comparison | 2 | 2 |
| 4 | 19 | 12.4 | 5 | 1 | 3 | 6 (16.9 s) | 4: route map panel, chart panel (not placed), chart full, date title | 1 | 2 |
| 5 | 20 | 13.0 | 3 | 1 | 2 | 7 (20.6 s) | 4: route map panel, 2 charts full, date title | 4 | 3 (art layer, now off) |

On-screen text items in pass 4, by kind:
- 1 route, 1 distance with unit, 2 place names, 4 chart bars with units, 1 date, 1 place label.
- No typed narration, in any pass.

Frame checks, opening:
- Blank frames: 41, then 8, then 47 (pass 5 regression, cause fixed, not re-rendered).
- Cut faces: 0 by detector. The viewer noted one tight head crop in pass 4.
- Repeats: 0.

## Cost and wall clock

| Project | Spend USD | Main items |
|---|---|---|
| alaska_v3_open (6 runs) | 3.51 | picker on anthropic_direct 0.97, AI prompt rewrites (editor) 0.57, judge checks 0.37 |
| alaska_v3 | 5.16 | director 2.40 and picker 2.36, both on anthropic_direct |
| dunkirk_v3 | 3.33 | director 1.19, picker on anthropic_direct |
| Night total | 12.00 | |

Opening pass wall clock:
- Pick 920 to 1050 s.
- Visuals 34 s, then 709 s and 2823 s while the art layer waited on a video provider (now off).
- Motion 22 to 57 s.
- Assembly 45 to 72 s.

## Verdict

- alaska_v3 against run 3: no verdict. alaska_v3 was not rendered. On the opening alone, pass 4 is better
  than night 2's alaska_v2b opening:
  - It opens on a real picture with a complete route-map panel on its words, where alaska_v2b had an empty map.
  - It has no typed narration.
  - Blank frames fell from about 20 s to 4 s.
- It is not yet better than run 3's opening. It still has a black gap, a wrong newspaper page and two
  AI-looking stills inside the first minute.
- dunkirk_v3 against dunkirk_base: no verdict. dunkirk_v3 was not rendered.

## What is still wrong, by screen time

1. Clauses with nothing usable in their pool (about 20 s of the opening). The setting fill is the answer but
   is untested on a render.
2. AI stills that look generated and do not agree with each other within a story (about 17 s).
3. Small pictures on a blurred fill held long (8 to 9 s each).
4. Panels that find no room over busy footage are dropped (6 s in pass 4).
5. Video cover crops do not yet keep heads whole.
6. Reply cache misses on reruns of the pick (cause not found). They make every pass cost about 0.5 USD
   instead of nothing.
