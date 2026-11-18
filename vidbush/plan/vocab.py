"""Names shared by the planner, the layouts lane and sourcing. Fixed: do not rename without the layouts lane."""
from __future__ import annotations

LAYOUT_KINDS = ("split_comparison", "number_over_footage", "name_card", "list_slideshow", "date_title",
                "route_map", "equals_card", "chart", "table", "timeline", "document_focus")
TREATMENTS = ("picture", "hold") + LAYOUT_KINDS
PRESENTED_KINDS = ("route_map", "chart", "table", "timeline", "document_focus")  # these take a presentation
DOC_PURPOSES = ("evidence", "period", "subject")  # why a document is on screen
MARKED_PURPOSES = ("evidence", "subject")  # these mark passages on the page
PRESENTATIONS = ("full", "panel", "full_then_dock", "panel_then_expand")
# layouts drawn over the clause's own picture: the edit picks that picture
OVER_PICTURE_KINDS = ("number_over_footage", "date_title")
REAL_KINDS = ("person", "place", "document", "object", "event")
GRAMMAR = ("sequence", "callback", "hero", "quiet")
MEDIA = ("moving", "still", "either")

TREATMENT_NOTES = {
    "picture": "one real picture (footage or photograph) of the subject; the default for most clauses",
    "hold": "keep the previous clause's picture or layout on screen across this clause (same shot)",
    "split_comparison": "two framed pictures side by side, each appearing on its own words; for a comparison "
                        "of two things the narration names",
    "number_over_footage": "a large number with its unit over darkened related footage, landing on the spoken "
                           "number; for one striking quantity",
    "name_card": "a real portrait with the person's name and a role line; only for a real named person",
    "list_slideshow": "one framed picture per spoken list item, sliding in on each item",
    "date_title": "a date or year as a title over related footage; for a jump in time",
    "route_map": "a map with places labelled as they are named and a line drawn between them; for a journey, "
                 "a distance or where something is",
    "equals_card": "two short text blocks joined by an equals sign; for a unit conversion or an equivalence",
    "chart": "bars or a line for two or more comparable numbers the narration gives",
    "table": "a small table of named items and their values, rows appearing as they are spoken",
    "timeline": "a line of dated events, each added as it is spoken; a later clause can return to it",
    "document_focus": "a real document, letter, newspaper page or record; how it is shown depends on why it is "
                      "there (purpose) and is never a static full page",
}

PRESENTATION_NOTES = {
    "full": "the graphic is what this clause is about and fills the frame",
    "panel": "the picture is the subject; the graphic sits as a small panel that orients the viewer",
    "full_then_dock": "the graphic is introduced full screen, then shrinks to a panel while the narration moves "
                      "on and it stays useful context",
    "panel_then_expand": "it sits as a panel, then grows to full screen when the narration turns to its detail; "
                         "stage 2 elements appear then",
}

# source classes the director writes queries for, and what each holds
SOURCE_CLASSES = {
    "stock": "stock video and photo libraries: modern, well shot, generic places, activities and objects; "
             "write a plain visual description (no names, no years)",
    "archive": "public archives and federal records (national archives, library of congress, internet archive, "
               "space agency): historical film, photographs, documents; write a proper name plus a year or decade",
    "encyclopedic": "encyclopedic media (wikimedia commons): photographs of named people, places, buildings, "
                    "objects, maps and documents; write the exact name of the thing",
    "web_video": "web video (youtube): news reports, event footage, documentaries, drone and travel films; "
                 "write the event or place as a person would title a video",
    "newspaper": "historical newspapers: front pages, headlines, articles; write the headline words or the "
                 "subject plus the newspaper name and year",
    "web_image": "general web image search: anything photographed and published, including specific signs, "
                 "buildings, documents and people; write the most specific name",
}

HOLD_MAX_S = 9.0  # longest a picture stays on screen in one shot (benchmark: nothing static past about 9 s)
LAYOUT_HOLD_MAX_S = 12.0  # maps, documents and text layouts may hold longer (benchmark up to about 11 s)
AI_HOLD_MAX_S = 4.0  # Adam: each AI still is on screen briefly
AI_RUN_MAX = 2  # Adam: no more than two AI stills in a row
READ_S_PER_WORD = 0.3
READ_MIN_S = 1.0
