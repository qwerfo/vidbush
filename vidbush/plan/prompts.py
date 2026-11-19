"""Prompts for the clause-first planner. General for any niche and script: every example is invented and
labelled as an example."""
from __future__ import annotations

import json
from typing import Any

from ..models import LAYOUT_ROLES
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
    "When the specific picture cannot be found, the editor fills the clause in this order: a real picture that "
    "fits the situation (same period, season, setting and circumstance), shown plain; a neighbouring real "
    "picture held a little longer; an AI still; a loosely related real picture. The narration's words are never "
    "typed over a picture in place of the missing one. An AI still is never used for a named real person or a "
    "document, so for those name the real_thing precisely and give queries that can find the real one.",
    "setting: when and where the clause happens, as the story tells it: period (a year, decade or 'present "
    "day'), season ('winter', 'summer', or 'any' when the story does not say), place (as precise as the script "
    "allows) and circumstance (one short phrase of what is going on, example, invented: 'families sheltering "
    "from a storm at night'). The editor refuses a picture whose period, season, setting or circumstance "
    "disagrees, because a viewer sees a summer pier under a winter storm as a mistake.",
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
    "layout with a simple picture. A layout is drawn only when all of it can be shown: every picture it needs, "
    "every place on a map, and at least 3 s on screen from its first anchored word; otherwise the clause keeps "
    "a plain picture, so plan layouts whose content exists.",
    "presentation is required for route_map, chart, table, timeline and document_focus, with a short "
    "presentation_reason. Panel is the default: when a map, chart, table, timeline or period document supports "
    "what is being said while a good real picture is on screen, it rides as a panel over that picture, placed "
    "by the picture, because the picture keeps the viewer in the scene while the graphic orients them. Full "
    "only when the graphic is itself the subject of the clause, or for a hero moment. Wherever the narration "
    "moves between named places, measures a distance or size, or compares quantities, plan a route_map, chart, "
    "table or timeline there (a panel over the picture by default), never a photograph of a map or chart: code "
    "draws them accurately from your elements, while a found map picture is usually the wrong map. full_then_dock "
    "when it is introduced full screen and stays useful as context while the narration moves on; "
    "panel_then_expand when it sits as a panel and grows to full screen when the narration turns to its detail "
    "(stage 2 elements appear then). Do not choose a position on screen.",
    "document_focus takes a presentation and a presentation_reason like the graphics, and a purpose in "
    "layout.purpose: evidence (a specific passage is quoted or referred to), period (the document is there for "
    "atmosphere or to place the story in time) or subject (the narration is about the document itself). How to "
    "choose, and why: a document proves something only when the viewer can see the words being quoted, so "
    "evidence goes in close and marks the passage; a document used for atmosphere should not stop the story, "
    "so it rides as a panel over footage; a full frame is for when the document is what the narration is about. "
    "A document is never a static full page: whenever it fills the frame, passages are marked as they are "
    "spoken. For evidence and subject give the passages to mark in spoken order as elements: text is the exact "
    "words expected on the page, anchor the narration words it is marked on. Example (invented): 'the council "
    "minutes record that the bridge was condemned in the spring' is evidence; the element text is 'bridge "
    "condemned' and its anchor 'the bridge was condemned'.",
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
    "screen, at most 6 words, and only what adds something the picture cannot: a number with its unit, a "
    "name, a date, a short label on a graphic, or (document_focus) a passage marked on its real page. Never a "
    "stretch of the narration typed out, because the viewer is already hearing it. A number carries its unit "
    "(example, invented: '40 KM', not '40'). "
    "picture_subject is the real picture the element needs, or empty. stage is 1, or 2 for detail that "
    "appears after a panel_then_expand. A timeline has a timeline_id; reuse that id in a later clause to "
    "return to the same timeline with its earlier events still on it.",
    "Each element must stay readable: it appears on its anchor words and needs 0.3 s per word, at least 1 s, "
    "before the shot ends; elements are listed in the order their anchors are spoken.",
    "grammar: a list, possibly empty, of labels you deliberately use: sequence (a whole view, then one closer "
    "view of the same subject or photograph on the next clause; one picture serves at most two shots in a "
    "row), callback (showing an earlier picture again because its meaning has changed, never within a minute "
    "of its last showing; set callback_to to that clause id), hero (one of "
    "two to four moments in the whole video that deserve the richest treatment), quiet (the music should "
    "drop away here before a reveal). Explain each label in grammar_note.",
    "Pictures that would be on screen for a long time or that the story returns to are worth a sequence or a "
    "callback; otherwise show something new on each clause.",
    "The first 60 seconds decide whether a viewer stays: the opening scene is the strongest stretch of the film. "
    "It opens on a strong real picture with motion or on a complete layout landing on its words, and no AI "
    "still is planned in its first 15 seconds.",
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
        "setting": {"period": "1925 | 1950s | present day", "season": "winter|spring|summer|autumn|any",
                    "place": "", "circumstance": "short phrase"},
        "anchor": "exact words from the clause",
        "real_thing": {"name": "", "kind": "person|place|document|object|event", "date": "", "place": ""},
        "queries": {"archive": ["Name 1920"], "stock": ["plain description"]},
        "layout": {"title": "", "timeline_id": "", "purpose": "evidence|period|subject (document_focus)",
                   "elements": [
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


# The layouts renderer reads these roles per kind (models.LAYOUT_ROLES) and these text formats.
ELEMENT_ROLES = {k.value: list(r) for k, r in LAYOUT_ROLES.items() if k.value not in ("document_focus",)}
ELEMENT_ROLES["document_focus"] = ["mark (a passage to mark: text is the words on the page)",
                                   "quote (the quoted words, shown alone when the page is not found)"]
ELEMENT_ROLES["formats"] = ["chart bar or point: '<label>: <value><unit>' (invented: 'Region A: 82%')",
                            "table header and row: cells separated by ' | '",
                            "timeline event: '<date>: <label>' (invented: '1925: First crossing')",
                            "route_map route: '<from place> to <to place>', optionally ' by <road|rail|river|sea|air|trail>'",
                            "route_map place: 'Town, State or Country' (invented: 'Riverton, Wyoming'), so the "
                            "map finds the right one; a map with a place it cannot find is not drawn",
                            "name_card: portrait is the picture element, name and role are text",
                            "number_over_footage and date_title: the background is the clause's own picture, "
                            "do not list it"]


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
        "element_roles": ELEMENT_ROLES,
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
