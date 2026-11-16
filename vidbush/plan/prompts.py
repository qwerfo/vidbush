"""Prompts for the clause-first planner. General for any niche and script: every example is invented and
labelled as an example."""
from __future__ import annotations

import json
from typing import Any

from . import vocab as V

DIRECTOR_SYSTEM = (
    "You are the director of a long-form documentary for YouTube. You plan what the viewer sees, clause by "
    "clause, for one scene at a time. Think in scenes: first decide what the viewer should discover, "
    "understand or feel in this scene, then choose for each clause the picture or layout that serves that, "
    "not merely one that matches its nouns. Reply with one JSON object only."
)

DIRECTOR_RULES = [
    "What your output is used for: every clause becomes a cut at the moment its first word is spoken. "
    "Researchers search the listed sources with your queries for the real_thing and the subject; an editor "
    "then compares the pictures found and picks one per clause, or answers that nothing fits. Layouts are "
    "drawn by code from your elements, each element appearing on its anchor words.",
    "When a real thing cannot be found, the editor goes down this ladder: a layout with real pictures inside, "
    "then a map, then text over related real footage, then an AI still. An AI still is never used for a named "
    "real person, a real event or a document, so for those name the real_thing precisely and give queries "
    "that can find the real one.",
    "scene_purpose: one sentence on what the viewer should discover, understand or feel in this scene.",
    "subject: one plain sentence of what the viewer should see, reflecting what the clause means in the "
    "story, not only its nouns. Example (invented): for 'the town paid the doctor in fish because it had no "
    "money' the subject is 'a doctor being handed a fish as payment', not 'a fish'.",
    "treatment: picture, hold, or one layout kind. hold keeps the previous clause's picture or layout on "
    "screen across this clause; use it to merge clauses that are about the same thing, and say why in "
    "reason. The first clause of a scene is never a hold. A shot (a clause plus its holds) must not run "
    f"over {V.HOLD_MAX_S:g} s for a picture or {V.LAYOUT_HOLD_MAX_S:g} s for a layout.",
    "Use a layout where it explains better than a picture: a comparison, a striking number, a person "
    "introduced by name, a spoken list, a jump in time, a journey or distance, a unit conversion, comparable "
    "numbers, a document the narration quotes. Most clauses stay pictures. Vary intensity: follow a dense "
    "layout with a simple picture.",
    "presentation is required for route_map, chart, table and timeline, with a short presentation_reason: "
    "full when the graphic is what the clause is about; panel when the picture is the subject and the graphic "
    "orients the viewer; full_then_dock when it is introduced full screen and stays useful as context while "
    "the narration moves on; panel_then_expand when it sits as a panel and grows to full screen when the "
    "narration turns to its detail (stage 2 elements appear then). Do not choose a position on screen.",
    "real_thing, when the clause names something that exists: name, kind (person, place, document, object, "
    "event), date and place, as precisely as the script allows. null otherwise.",
    "medium: moving when the subject is an action or a place best seen in motion, still when it is a "
    "photograph, portrait, document or object, either when both serve.",
    "anchor: the exact words of this clause the picture lands on, copied from the clause.",
    "queries: an object with source classes as keys and 1 to 3 queries each, written for that class (an "
    "archive wants a name and a year; a stock library wants a plain description). Only the classes that can "
    "hold this subject. A hold has no queries.",
    "layout, for a layout treatment: title where the layout has one, and elements in spoken order, each "
    "{role, text, anchor, picture_subject, stage}. anchor is the exact narration words the element appears "
    "on; they must be spoken inside this clause or the hold clauses that follow it. text is what is typed on "
    "screen, at most 6 words; it may use the narration's own key words when it is a number, a name, a date or "
    "a short phrase typed as it is said. A number carries its unit (example, invented: '40 KM', not '40'). "
    "picture_subject is the real picture the element needs, or empty. stage is 1, or 2 for detail that "
    "appears after a panel_then_expand. A timeline has a timeline_id; reuse that id in a later clause to "
    "return to the same timeline with its earlier events still on it.",
    "Each element must stay readable: it appears on its anchor words and needs 0.3 s per word, at least 1 s, "
    "before the shot ends; elements are listed in the order their anchors are spoken.",
    "grammar: a list, possibly empty, of labels you deliberately use: sequence (wide, then medium, then "
    "detail of the same subject or the same photograph across consecutive clauses), callback (showing an "
    "earlier picture again because its meaning has changed; set callback_to to that clause id), hero (one of "
    "two to four moments in the whole video that deserve the richest treatment), quiet (the music should "
    "drop away here before a reveal). Explain each label in grammar_note.",
    "Pictures that would be on screen for a long time or that the story returns to are worth a sequence or a "
    "callback; otherwise show something new on each clause.",
    "No em dash characters anywhere. Keep every string short and concrete.",
]

OUTPUT_SHAPE = {
    "scene_purpose": "one sentence",
    "clauses": [{
        "id": "c001", "subject": "one plain sentence", "treatment": "picture|hold|<layout kind>",
        "reason": "why hold or why this layout; empty for a plain picture",
        "presentation": "full|panel|full_then_dock|panel_then_expand (route_map, chart, table, timeline only)",
        "presentation_reason": "short",
        "medium": "moving|still|either",
        "anchor": "exact words from the clause",
        "real_thing": {"name": "", "kind": "person|place|document|object|event", "date": "", "place": ""},
        "queries": {"archive": ["Name 1920"], "stock": ["plain description"]},
        "layout": {"title": "", "timeline_id": "", "elements": [
            {"role": "", "text": "", "anchor": "", "picture_subject": "", "stage": 1}]},
        "grammar": [], "grammar_note": "", "callback_to": "",
    }],
}


def full_script_text(script) -> str:
    parts = []
    for s in script.scenes:
        parts.append(f"[{s.id} {s.title} ({s.purpose.value})]")
        parts.append(" ".join(b.text for b in s.beats))
    return "\n".join(parts)


def director_user(script, scene, clauses, earlier: list[dict[str, Any]], heroes_so_far: int,
                  timelines: list[str]) -> str:
    body = {
        "video_title": script.title,
        "full_script": full_script_text(script),
        "this_scene": {"id": scene.id, "title": scene.title, "purpose": scene.purpose.value},
        "clauses": [{"id": c.id, "start": round(c.start, 2), "seconds": c.duration, "text": c.text}
                    for c in clauses],
        "sources": V.SOURCE_CLASSES,
        "treatments": V.TREATMENT_NOTES,
        "presentations": V.PRESENTATION_NOTES,
        "earlier_scenes_showed": earlier,
        "hero_moments_used_so_far": heroes_so_far,
        "timeline_ids_so_far": timelines,
        "rules": DIRECTOR_RULES,
        "output": OUTPUT_SHAPE,
    }
    return json.dumps(body, ensure_ascii=False)


def director_retry(user: str, previous: Any, errors: list[str]) -> str:
    return (user + "\n\nYour previous reply is below, followed by the problems found in it. Return the whole "
            "corrected JSON object, every clause again, fixing each problem.\n\nPrevious reply:\n"
            + json.dumps(previous, ensure_ascii=False)[:24000] + "\n\nProblems:\n"
            + "\n".join(f"- {e}" for e in errors))


EDIT_SYSTEM = (
    "You are the picture editor of a documentary. You see the clauses of one scene with what each should "
    "show, and numbered contact sheets of the pictures found for the scene. You assign pictures to clauses by "
    "comparing them side by side. Reply with one JSON object only."
)

EDIT_RULES = [
    "What your output is used for: each pick is cut in at the clause's first word and plays from the frame "
    "you name. A clause you leave without a pick goes down the fallback ladder described below.",
    "Compare the candidates against the clause subject: what the picture shows, when and where. A picture "
    "is right when a viewer hearing the clause would see the thing meant. Prefer the real thing named in "
    "real_thing over a look-alike. Use the titles, sources and dates given; they are often more reliable "
    "than the frame for identity.",
    "Answer nothing fits rather than force a weak picture: set pick to null and say why in reason.",
    "One candidate may cover two adjacent clauses (set the same pick). Using a candidate again anywhere "
    "else is allowed only as a deliberate callback: set callback true and say what changed.",
    "frame: the letter of the keyframe to start from (a, b or c for video, a for a photograph). Choose the "
    "frame that shows the subject; the shot plays from there.",
    "label: a place or name label for the picture, only when the picked picture is the real thing it "
    "names (a photograph of the actual building, the actual person). Otherwise empty.",
    "AI stills are marked AI in the sheet. An AI still must show what the clause means, not only its "
    "nouns (example, invented: a line about villagers pooling money to buy a museum's stuffed bear needs "
    "coins and a stuffed bear in a glass case, not people offering coins to a live bear). Never pick an AI "
    "still for a named real person, a real event or a document. At most two AI stills in a row, each "
    f"on screen at most {V.AI_HOLD_MAX_S:g} s. Where a stretch has no real picture, go up the ladder before "
    "a third AI still: ask for a layout, a map, hold the previous real picture longer (pick it again on "
    "the next clause), or text over related real footage.",
    "fallback, for a clause with no pick: one of map, hold_previous, text_over_footage, ai_still. map: a "
    "map of the place the clause is about (set map_place). text_over_footage: a short title over related real "
    "footage (set fallback_pick to that candidate and overlay_text to the words, at most 6). Never an AI still "
    "for fallback_pick.",
    "For layout clauses, pick pictures for the layout elements that have a picture_subject (element_picks), "
    "and for over-picture layouts and panels also the picture beneath (pick).",
    "No em dash characters anywhere.",
]

EDIT_OUTPUT = {
    "picks": [{"clause": "c001", "pick": 12, "frame": "b", "label": "", "callback": False,
               "reason": "short", "fallback": "", "fallback_pick": None, "overlay_text": "", "map_place": "",
               "element_picks": [{"element": 0, "pick": 7, "frame": "a"}]}],
}


def edit_prompt(scene_purpose: str, clauses: list[dict[str, Any]], catalogue: list[dict[str, Any]],
                sheet_note: str) -> str:
    body = {"scene_purpose": scene_purpose, "clauses": clauses, "candidates": catalogue,
            "sheets": sheet_note, "rules": EDIT_RULES, "output": EDIT_OUTPUT}
    return json.dumps(body, ensure_ascii=False)


def edit_retry(prompt: str, previous: Any, errors: list[str]) -> str:
    return (prompt + "\n\nYour previous reply is below, followed by the problems found in it. Return the "
            "whole corrected JSON object, every clause again.\n\nPrevious reply:\n"
            + json.dumps(previous, ensure_ascii=False)[:16000] + "\n\nProblems:\n"
            + "\n".join(f"- {e}" for e in errors))
