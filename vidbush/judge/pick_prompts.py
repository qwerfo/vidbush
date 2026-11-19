"""Prompts for the picker (describe once, pick by comparison) and for the ladder's re-direct.

General for any niche and script: every example is invented and labelled as an example. Each principle
carries its reason, because a model applies a rule it understands more consistently than a bare order.
"""
from __future__ import annotations

import json
from typing import Any

from ..plan import vocab as V

PICK_SYSTEM = (
    "You are the picture editor of a documentary. For one scene you get its purpose, each clause with what "
    "it should show, and the pool of assets found for the scene: a written record of each asset and a "
    "numbered image of each. You assign pictures to clauses by comparing the assets with each other. "
    "Reply with one JSON object only."
)

PICK_RULES = [
    "What your output is used for: each pick is cut in at the clause's first word and plays from the start "
    "time you give, shown plain: no words are ever typed over it. A clause you leave without a pick is filled "
    "in this order: your fitting picture, a complete layout, a neighbouring picture held a little longer, an AI "
    "still (only where allowed), your loose picture. So saying nothing fits is cheap; a wrong picture is not.",
    "pick: the asset that shows the thing the clause is about at the moment it is said, because the viewer "
    "hears the words and looks for them on screen. Example (invented): for 'the mill closed in the winter of "
    "the flood' a picture of the mill closed or of the flood is the pick.",
    "Fit means the situation, not a shared noun. Each clause has a setting (period, season, place, "
    "circumstance) and each asset a record (shows, place, period, look). An asset fits a clause only when "
    "period, season, setting and circumstance agree with what is being said; refuse a mismatch, because the "
    "viewer sees a summer street under a line about a winter storm, or a modern crowd under a line about a "
    "1920s celebration, as a mistake. A modern picture may cover a historical clause only when its subject is "
    "unchanged by time (a landscape, a mountain, a coastline, the open sea). Sharing a word with the "
    "narration is not fitting.",
    "fitting: for a clause with no pick, and for every full layout clause (in case the layout cannot be "
    "drawn), the asset that best fits the situation described, by the rule above, though it does not show "
    "the specific thing. Example (invented): for 'in the winter of 1890 the miners slept in tents on the ice' "
    "a black and white photograph of a tent camp in snow fits; a modern summer campsite does not. Null when "
    "nothing fits.",
    "Never pick, as pick, fitting or loose, a clip from a feature film or television drama (recognisable "
    "actors, a film's staging), a picture whose main content is a small sign, plaque or page the viewer cannot "
    "read full screen, or a picture with a burned-in title, channel name or caption.",
    "hold: for a clause with no pick, 'previous' or 'next' when that neighbouring clause's picture also fits "
    "this clause's situation and could stay on screen across it, else empty.",
    "loose: for every clause with neither pick nor fitting (even when a hold or an AI still may fill it), the "
    "real asset that best keeps the viewer in the "
    "story: the same place, the same people or the same subject, shown plain while the narration explains. "
    "A landscape or street of the story's region is a good loose picture for almost any line about it. Null "
    "only when every asset would show the viewer something false (a different named place or person, or "
    "modern objects under a historical line); a clause with nothing at all is far worse than a loose picture.",
    "A named real thing (the clause's real_thing) is only shown by an asset whose record supports that it is "
    "that thing: its title, description, page text, date, or the text visible in the frame. A look-alike is "
    "refused, because a viewer who knows the subject sees the substitution and stops trusting the film. For "
    "every pick of a clause with a real_thing, copy into identity the exact words of the record that show "
    "it is that thing, and name the record field they come from (title, description, page_url, date, "
    "text_in_frame, ocr_text). Code checks that the words are in that field.",
    "The moment matters as much as the name: a picture of the named place or object in another period "
    "(a modern photograph of a place the clause shows in a war, a ruin for a working building) is related, "
    "not the thing at the moment it is said, because the viewer sees the wrong time. Use the record's period, "
    "look and date. Example (invented): for 'the pier was crowded with refugees in 1912' a recent photo of "
    "the empty pier is related.",
    "A document is only accepted as evidence when the quoted words were found on the page: a page that only "
    "looks similar would put the marks on words that are not there. For a document_focus with passages, "
    "pick a document only when its ocr_text or text_in_frame contains each passage; code checks this.",
    "Between two assets that both fit, prefer the one with an open licence (public domain or Creative "
    "Commons) because it can be published safely; then the sharper one; then the one that has not already "
    "been near this subject in the film, so the film keeps showing new things.",
    "Never pick an asset whose record lists a watermark, a stock preview mark, burned-in subtitles or a "
    "channel logo (they are removed from the pool before you see it; if one slips through, refuse it).",
    "start: for a video, the second in the asset where the shot begins, inside one of the record's usable "
    "ranges, so that start plus the shot length stays inside the clip. The record's frame times tell you what "
    "plays when. For a still, leave start null.",
    "One asset may cover at most two shots in a row (a whole view, then one closer view): never three. Using "
    "an asset again anywhere else is allowed only as a deliberate callback at least 60 s after its last "
    "showing: set callback true and say what changed.",
    "The previous scene's last picks are given so the cut into this scene reads as continuous: do not open "
    "with the same picture, and prefer a first picture that follows on from them.",
    f"AI stills are marked ai. Never pick one for a named real person or a document, and never in the first "
    f"{V.AI_OPENING_S:g} s of the film. At most {V.AI_RUN_MAX} AI stills in a row, each on screen at most "
    f"{V.AI_HOLD_MAX_S:g} s. Never give an AI still as fitting or loose.",
    "label: a place, name or date label, only when the picked asset's record supports it (the label words "
    "appear in its title, description, date, page text or visible text). Give label_source, the record "
    "field it comes from. Otherwise empty: a label is never written from what the clause hoped to show.",
    "For a clause with no pick: set pick null, say in reason what the pool has and what is missing, and give "
    "search_again: 1 to 3 queries that could find the missing picture, written from what the pool did "
    "contain and what it lacked (example, invented: the pool has the modern bridge but no photograph of its "
    "opening, so 'bridge opening ceremony 1931 photograph').",
    "For layout clauses, pick pictures for the elements that have a picture_subject (element_picks), and "
    "for over-picture layouts and panels the picture beneath (pick). For a document_focus, document_pick is "
    "the document itself, never AI; pick is the footage beneath it unless its presentation is full.",
    "No em dash characters anywhere.",
]

PICK_OUTPUT = {
    "picks": [{"clause": "c001", "pick": 12, "start": 3.5, "reason": "short", "identity": "",
               "identity_field": "", "label": "", "label_source": "", "callback": False,
               "search_again": [], "element_picks": [{"element": 0, "pick": 7, "start": None}],
               "document_pick": None, "fitting": None, "fitting_start": None, "hold": "",
               "loose": None, "loose_start": None}],
}


def pick_prompt(scene_purpose: str, clauses: list[dict[str, Any]], records: list[dict[str, Any]],
                previous: list[dict[str, Any]], image_note: str) -> str:
    body = {"scene_purpose": scene_purpose, "clauses": clauses, "assets": records,
            "previous_scene_last_picks": previous, "images": image_note, "rules": PICK_RULES,
            "output": PICK_OUTPUT}
    return json.dumps(body, ensure_ascii=False)


def pick_retry(prompt: str, previous: Any, errors: list[str]) -> str:
    return (prompt + "\n\nYour previous reply is below, followed by the problems code found in it. Return the "
            "whole corrected JSON object, every clause again, fixing each problem.\n\nPrevious reply:\n"
            + json.dumps(previous, ensure_ascii=False)[:16000] + "\n\nProblems:\n"
            + "\n".join(f"- {e}" for e in errors))


RESTATE_SYSTEM = (
    "You are the director of a documentary. The editor found no picture for some clauses. You restate each "
    "one as a layout that needs no missing picture. Reply with one JSON object only."
)

RESTATE_KINDS = {
    "number_over_footage": "a number from the clause, with its unit, over real footage we have",
    "split_comparison": "two things we do have pictures of, side by side",
    "date_title": "a date or period title over related real footage",
    "route_map": "a map of the places or the route the clause names",
}

RESTATE_RULES = [
    "What your output is used for: code draws the layout from your fields over the asset you name, so the "
    "clause states its point without the picture that could not be found. A layout that needs a picture we "
    "lack fails again.",
    "Choose kind from the list given. Use only assets from the pool list, by number, and only ones whose "
    "record shows real footage or a real photograph (never an AI still beneath a named real person, event "
    "or document).",
    "The asset beneath must not contradict the clause's place or period (no modern beach under a past "
    "event), because the typed words then sit on a picture that says something else.",
    "text: what is typed on screen, at most 6 words: only a number with its unit, a date, or short labels; "
    "never a stretch of the narration typed out. anchor: the exact clause words the text appears on.",
    "A layout is drawn only when it is complete: both pictures of a comparison, every place of a map, the "
    "picture beneath. When it cannot be complete, set kind null.",
    "For route_map give places, the place names in the order they are said. For split_comparison give "
    "pictures, two asset numbers, and text as 'left words / right words'. For the other kinds give beneath, "
    "the asset the text sits over.",
    "When no layout can state the clause honestly with what we have, set kind null and say why.",
    "No em dash characters anywhere.",
]

RESTATE_OUTPUT = {"restated": [{"clause": "c001", "kind": "text_over_footage", "text": "", "anchor": "",
                                "beneath": 4, "pictures": [], "places": [], "reason": "short"}]}


def restate_prompt(scene_purpose: str, clauses: list[dict[str, Any]], pool: list[dict[str, Any]]) -> str:
    return json.dumps({"scene_purpose": scene_purpose, "clauses_with_nothing": clauses, "pool": pool,
                       "kinds": RESTATE_KINDS, "rules": RESTATE_RULES, "output": RESTATE_OUTPUT},
                      ensure_ascii=False)
