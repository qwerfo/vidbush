"""Vidbush data contracts.

Every stage reads and writes these models as JSON on disk. They are the stable
interface between modules. Change them only with a matching schema bump.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"


def new_id(prefix: str) -> str:
    """Short unique id with a readable prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


class Model(BaseModel):
    """Base model: tolerant of extra keys so older files still load."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


# ---------------------------------------------------------------- enums

class InputMode(str, Enum):
    brief = "brief"
    script = "script"
    voiceover = "voiceover"


class VideoFormat(str, Enum):
    documentary = "documentary"
    top10 = "top10"


class ShotType(str, Enum):
    footage = "footage"
    footage_overlay = "footage_overlay"
    image = "image"
    image_overlay = "image_overlay"
    ai_image = "ai_image"
    ai_video = "ai_video"
    motion_graphic = "motion_graphic"


SHOT_TYPES: list[str] = [t.value for t in ShotType]
FOOTAGE_TYPES = {ShotType.footage, ShotType.footage_overlay}
STILL_TYPES = {ShotType.image, ShotType.image_overlay, ShotType.ai_image}
OVERLAY_TYPES = {ShotType.footage_overlay, ShotType.image_overlay}


class ScenePurpose(str, Enum):
    hook = "hook"
    setup = "setup"
    act = "act"
    item = "item"
    turn = "turn"
    climax = "climax"
    resolution = "resolution"
    outro = "outro"


class Emphasis(str, Enum):
    low = "low"
    normal = "normal"
    high = "high"


class SourceKind(str, Enum):
    youtube = "youtube"
    wikimedia = "wikimedia"
    archive = "archive"
    loc = "loc"
    nasa = "nasa"
    pexels = "pexels"
    pixabay = "pixabay"
    local = "local"  # operator footage library folder (and bundled fixtures)
    ai_image = "ai_image"
    ai_video = "ai_video"
    upload = "upload"
    motion = "motion"  # rendered motion graphic


class StageStatus(str, Enum):
    pending = "pending"
    running = "running"
    done = "done"
    failed = "failed"
    skipped = "skipped"


STAGES: list[str] = [
    "intake",
    "research",
    "outline",
    "script",
    "narration",
    "shotlist",
    "sourcing",
    "judge",
    "visuals",
    "motion",
    "mix",
    "assembly",
    "packaging",
]


# ---------------------------------------------------------------- project

class ProjectPaths(Model):
    root: str  # output/<project_id>
    work: str  # output/<project_id>/work
    sources: str  # output/<project_id>/sources
    segments: str  # output/<project_id>/work/segments
    thumbnails: str  # output/<project_id>/thumbnails


class StageRecord(Model):
    name: str
    status: StageStatus = StageStatus.pending
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    attempts: int = 0
    outputs: list[str] = Field(default_factory=list)


class Brief(Model):
    topic: str
    target_minutes: float = 8.0
    format: VideoFormat = VideoFormat.documentary
    audience: str = ""
    angle: str = ""


class Project(Model):
    schema_version: str = SCHEMA_VERSION
    id: str
    created_at: float = Field(default_factory=time.time)
    brand_id: str = "default"
    input_mode: InputMode
    format: VideoFormat = VideoFormat.documentary
    target_minutes: float = 8.0
    title: str = ""
    brief: Optional[Brief] = None
    script_path: Optional[str] = None  # input script (script mode or voiceover sidecar)
    voiceover_path: Optional[str] = None  # input audio (voiceover mode)
    status: Literal["created", "running", "done", "needs_review", "failed"] = "created"
    current_stage: Optional[str] = None
    stages: dict[str, StageRecord] = Field(default_factory=dict)
    paths: ProjectPaths
    style_profile_id: Optional[str] = None  # overrides brand default
    seed: int = 0  # bounded randomisation seed, fixed per project
    status_note: str = ""  # why the status is what it is, e.g. the cost cap reason (jobs/costcap.py)


# ---------------------------------------------------------------- research

class FactCard(Model):
    id: str
    claim: str
    source_url: str = ""
    source_title: str = ""
    confidence: float = 0.7
    date: str = ""  # date or era the fact refers to
    status: Literal["verified", "unverified", "contradicted"] = "unverified"
    numbers: list[dict[str, Any]] = Field(default_factory=list)  # [{value, unit, label}]


class VisualAnchor(Model):
    kind: Literal["place", "date", "object", "person", "event", "number"]
    name: str
    detail: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    fact_ids: list[str] = Field(default_factory=list)


class Research(Model):
    topic: str
    facts: list[FactCard] = Field(default_factory=list)
    anchors: list[VisualAnchor] = Field(default_factory=list)
    sources: list[dict[str, str]] = Field(default_factory=list)  # [{url, title}]


# ---------------------------------------------------------------- script

class OutlineSection(Model):
    id: str
    title: str
    purpose: ScenePurpose
    summary: str
    target_words: int
    fact_ids: list[str] = Field(default_factory=list)
    item_number: Optional[int] = None  # top10 countdown number


class Outline(Model):
    format: VideoFormat
    title: str
    sections: list[OutlineSection]


class Beat(Model):
    id: str
    text: str  # one to three sentences of narration
    facts_used: list[str] = Field(default_factory=list)
    emphasis: Emphasis = Emphasis.normal


class Scene(Model):
    id: str
    title: str
    purpose: ScenePurpose
    beats: list[Beat]
    item_number: Optional[int] = None


class Script(Model):
    title: str = ""
    format: VideoFormat = VideoFormat.documentary
    scenes: list[Scene]

    def beats(self) -> list[Beat]:
        return [b for s in self.scenes for b in s.beats]

    def word_count(self) -> int:
        return sum(len(b.text.split()) for b in self.beats())


# ---------------------------------------------------------------- narration

class Word(Model):
    text: str
    start: float
    end: float
    beat_id: str = ""


class Narration(Model):
    audio_path: str  # normalised narration wav (-16 LUFS)
    duration: float
    words: list[Word]
    provider: str = "mock"
    aligned_by: Literal["provider", "whisper", "silence", "estimate"] = "estimate"
    beat_spans: dict[str, list[float]] = Field(default_factory=dict)  # beat_id -> [start, end]


# ---------------------------------------------------------------- motion spec

class MotionKind(str, Enum):
    map = "map"
    bar_chart = "bar_chart"
    line_chart = "line_chart"
    number_card = "number_card"
    title_card = "title_card"
    lower_third = "lower_third"
    callout = "callout"
    highlight_box = "highlight_box"
    text_overlay = "text_overlay"
    arrow = "arrow"
    recap_grid = "recap_grid"


class StyleTokens(Model):
    font: str = "Inter"
    palette: str = "documentary_warm"


class MapMarker(Model):
    lat: float
    lon: float
    label: str = ""
    appear_at: float = 0.0  # seconds into the clip


class MapRoute(Model):
    points: list[list[float]]  # [[lat, lon], ...]
    reveal_start: float = 0.5
    reveal_end: float = 3.0
    label: str = ""
    dashed: bool = False


class MapArrow(Model):
    start: list[float]  # [lat, lon]
    end: list[float]
    appear_at: float = 1.0
    label: str = ""


class MapLabel(Model):
    lat: float
    lon: float
    text: str
    size: Literal["small", "medium", "large"] = "medium"


class MapPayload(Model):
    center: list[float]  # [lat, lon]
    zoom: float = 3.0  # 1 = whole world, each step doubles scale
    zoom_end: Optional[float] = None  # animated push-in target
    markers: list[MapMarker] = Field(default_factory=list)
    routes: list[MapRoute] = Field(default_factory=list)
    arrows: list[MapArrow] = Field(default_factory=list)
    labels: list[MapLabel] = Field(default_factory=list)
    shaded_regions: list[str] = Field(default_factory=list)  # country names or ISO A3
    title: str = ""


class ChartSeries(Model):
    name: str = ""
    values: list[float]


class ChartPayload(Model):
    title: str = ""
    labels: list[str]
    series: list[ChartSeries]
    units: str = ""
    animate: Literal["grow", "draw", "count_up"] = "grow"
    highlight_index: Optional[int] = None


class NumberCardPayload(Model):
    number: str  # "10", "#3", "1914", "$375M"
    label: str = ""
    sublabel: str = ""
    count_up: bool = False


class TextPayload(Model):
    """Payload for title_card, lower_third, callout, text_overlay."""

    title: str
    subtitle: str = ""
    position: Literal["center", "lower_left", "upper_left", "upper_right", "lower_right"] = "center"


class BoxPayload(Model):
    """Payload for highlight_box and arrow overlays. Coordinates are 0..1 of frame."""

    x: float = 0.3
    y: float = 0.3
    w: float = 0.4
    h: float = 0.3
    label: str = ""
    to_x: Optional[float] = None  # arrow head target
    to_y: Optional[float] = None


class RecapPayload(Model):
    title: str = "Recap"
    items: list[str]  # ordered labels, item 10 first or item 1 first as given
    numbers: Optional[list[int]] = None  # explicit item numbers (top10 countdown); highlight number 1


class MotionSpec(Model):
    kind: MotionKind
    duration: float
    width: int = 1920
    height: int = 1080
    fps: int = 30
    background: Literal["transparent", "solid", "footage_behind"] = "solid"
    style: StyleTokens = Field(default_factory=StyleTokens)
    source_note: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    anchor: str = ""  # text overlays: the narration words it belongs to (director seat; else its title is matched)
    start_at: float = 0.0  # text overlays: seconds into the shot it appears, just after its anchor is spoken

    def typed_payload(self) -> BaseModel:
        """Parse payload into the model for this kind."""
        return PAYLOAD_MODELS[self.kind](**self.payload)


PAYLOAD_MODELS: dict[MotionKind, type[BaseModel]] = {
    MotionKind.map: MapPayload,
    MotionKind.bar_chart: ChartPayload,
    MotionKind.line_chart: ChartPayload,
    MotionKind.number_card: NumberCardPayload,
    MotionKind.title_card: TextPayload,
    MotionKind.lower_third: TextPayload,
    MotionKind.callout: TextPayload,
    MotionKind.text_overlay: TextPayload,
    MotionKind.highlight_box: BoxPayload,
    MotionKind.arrow: BoxPayload,
    MotionKind.recap_grid: RecapPayload,
}


# ---------------------------------------------------------------- shots

class Candidate(Model):
    id: str
    source: SourceKind
    url: str = ""
    local_path: str = ""
    media: Literal["video", "image"] = "image"
    in_point: float = 0.0
    out_point: float = 0.0
    width: int = 0
    height: int = 0
    licence: str = "unknown"
    author: str = ""
    attribution: str = ""
    title: str = ""
    keyframes: list[str] = Field(default_factory=list)
    score: Optional[float] = None
    judge_reason: str = ""
    reject_reason: Optional[str] = None
    flags: list[str] = Field(default_factory=list)  # nc_nd, mock, judge_mock, low_res, anachronism, depthflow, fixture_rotation, frame_grab
    crop: Optional[list[float]] = None  # derived framing [x, y, w, h] in 0..1


class Shot(Model):
    id: str
    index: int = 0
    beat_id: str
    scene_id: str = ""
    start: float
    end: float
    duration: float
    type: ShotType
    intent: str
    must_show: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    overlay: Optional[MotionSpec] = None
    motion: Optional[MotionSpec] = None  # full-frame spec for motion_graphic shots
    ai_prompt: str = ""
    candidates: list[Candidate] = Field(default_factory=list)
    selected: Optional[str] = None
    judge_score: Optional[float] = None
    judge_notes: str = ""
    attempts: int = 0
    flags: list[str] = Field(default_factory=list)
    framing: Literal["full", "tight", "detail"] = "full"
    transition_in: Literal["cut", "fade"] = "cut"
    overlay_path: str = ""  # rendered alpha clip for overlay (set by the motion stage)
    kenburns: Optional[dict[str, float]] = None  # optional explicit {zoom0, zoom1, x0, y0, x1, y1}
    render_segment_path: str = ""
    render_hash: str = ""

    def selected_candidate(self) -> Optional[Candidate]:
        for c in self.candidates:
            if c.id == self.selected:
                return c
        return None


class ShotList(Model):
    schema_version: str = SCHEMA_VERSION
    project_id: str
    fps: int = 30
    width: int = 1920
    height: int = 1080
    duration: float
    shots: list[Shot]

    def by_id(self, shot_id: str) -> Shot:
        for s in self.shots:
            if s.id == shot_id:
                return s
        raise KeyError(shot_id)

    def reindex(self) -> None:
        """Recompute index, start and end from durations in order."""
        t = 0.0
        for i, s in enumerate(self.shots):
            s.index = i
            s.start = round(t, 3)
            t += s.duration
            s.end = round(t, 3)
        self.duration = round(t, 3)


class VisualIdea(Model):
    """Director output per beat, before slot assignment.

    family picks which slot types it can fill: footage -> footage/footage_overlay,
    still -> image/image_overlay/ai_image, motion -> motion_graphic.
    """

    beat_id: str
    family: Literal["footage", "still", "motion"]
    intent: str
    must_show: list[str] = Field(default_factory=list)
    must_avoid: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    motion: Optional[MotionSpec] = None  # required when family == motion
    overlay: Optional[MotionSpec] = None  # optional overlay for footage or still
    priority: float = 1.0  # higher is used first
    opening_safe: bool = True  # has real activity, ok for the dense opening


class DirectorPlan(Model):
    ideas: list[VisualIdea]


# ---------------------------------------------------------------- style and brand

class TypePacing(Model):
    min: float
    max: float
    mean: float


class PacingProfile(Model):
    footage: TypePacing = TypePacing(min=3, max=7, mean=4.8)
    footage_overlay: TypePacing = TypePacing(min=3, max=7, mean=5.0)
    image: TypePacing = TypePacing(min=4, max=8, mean=5.5)
    image_overlay: TypePacing = TypePacing(min=4, max=8, mean=5.8)
    ai_image: TypePacing = TypePacing(min=4, max=8, mean=5.5)
    ai_video: TypePacing = TypePacing(min=3, max=6, mean=4.5)
    motion_graphic: TypePacing = TypePacing(min=6, max=15, mean=8.5)
    hook_seconds: float = 60.0
    hook_multiplier: float = 0.7
    jitter: float = 0.25  # bounded randomisation, fraction of mean
    # At most one full screen card per this many seconds outside the hook and the recap (0 turns it off).
    # Reference channels keep the picture on footage and stills and move maps and charts over it, so a
    # card is a punctuation mark, not a slideshow.
    card_spacing_s: float = 45.0

    def for_type(self, t: ShotType | str) -> TypePacing:
        return getattr(self, ShotType(t).value)


class CaptionStyle(Model):
    enabled: bool = True
    font: str = "Inter"
    weight: Literal["Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black"] = "ExtraBold"
    size: int = 64
    color: str = "#FFFFFF"
    outline_color: str = "#000000"
    outline: int = 4
    shadow: int = 0
    highlight_color: str = "#FFD84A"
    uppercase: bool = False
    min_words: int = 3
    max_words: int = 5
    position: Literal["bottom_band"] = "bottom_band"
    band_height: float = 0.5  # fraction of the bottom band (or bottom margin) from top of band


class Letterbox(Model):
    enabled: bool = True
    ratio: float = 2.39


class Transitions(Model):
    cut_probability: float = 0.9
    crossfade_duration: float = 0.4  # used as fade-through-black length at scene starts


class ColourGrade(Model):
    lut_path: Optional[str] = None


class MusicStyle(Model):
    ducking_depth_db: float = 12.0
    attack_ms: float = 80.0
    release_ms: float = 600.0
    bed_gain_db: float = -18.0
    scene_crossfade_s: float = 1.5


class MotionStyle(Model):
    ken_burns_intensity: float = 0.12  # max zoom delta fraction
    depthflow: bool = False


CARDS_KEY = "full_screen_cards"
OVERLAYS_KEY = "animated_overlays"
PICTURE_KEYS = ("footage", "footage_overlay", "image", "image_overlay", "ai_image", "ai_video")


def default_visual_mix() -> dict[str, float]:
    """Documentary default from a reference analysis of a top geography channel (share of shot count).

    The reference measured 42 percent animation. Most of it is maps, charts and callouts moving over
    footage and stills, so it is split: 12 full screen cards and 30 animated overlays that decorate a
    footage or image shot (counted on top of the shot they decorate, see StyleProfile.shot_mix).
    """
    return {"footage": 20, "footage_overlay": 21, "image": 7, "image_overlay": 10, CARDS_KEY: 12,
            OVERLAYS_KEY: 30, "ai_image": 0, "ai_video": 0}


class StyleProfile(Model):
    id: str
    name: str = ""
    visual_mix: dict[str, float] = Field(default_factory=default_visual_mix)
    pacing: PacingProfile = Field(default_factory=PacingProfile)
    captions: CaptionStyle = Field(default_factory=CaptionStyle)
    letterbox: Letterbox = Field(default_factory=Letterbox)
    transitions: Transitions = Field(default_factory=Transitions)
    colour: ColourGrade = Field(default_factory=ColourGrade)
    music: MusicStyle = Field(default_factory=MusicStyle)
    motion: MotionStyle = Field(default_factory=MotionStyle)
    palette: str = "documentary_warm"
    top10: bool = False  # number cards, item lower thirds, recap grid
    provenance: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _legacy_mix(self) -> "StyleProfile":
        """Old profiles named full screen cards motion_graphic; read that key as full_screen_cards."""
        mix = self.visual_mix
        if "motion_graphic" in mix:
            legacy = float(mix.pop("motion_graphic") or 0)
            mix.setdefault(CARDS_KEY, legacy)
        return self

    def shot_mix(self) -> dict[str, float]:
        """Target share of shot count per ShotType value, in percent, summing to 100.

        full_screen_cards become motion_graphic shots. animated_overlays are not shots of their own: they
        decorate footage or image shots, so they are added to footage_overlay and image_overlay in the
        ratio of footage to still shots in the picture mix (ai types excluded from that ratio).
        """
        mix = {k: max(0.0, float(v or 0)) for k, v in self.visual_mix.items()}
        out = {k: mix.get(k, 0.0) for k in PICTURE_KEYS}
        out["motion_graphic"] = mix.get(CARDS_KEY, 0.0)
        extra = mix.get(OVERLAYS_KEY, 0.0)
        foot = out["footage"] + out["footage_overlay"]
        still = out["image"] + out["image_overlay"]
        share = foot / (foot + still) if foot + still > 0 else 1.0
        out["footage_overlay"] += extra * share
        out["image_overlay"] += extra * (1.0 - share)
        total = sum(out.values())
        if total <= 0:
            return {"footage": 100.0}
        return {k: round(100.0 * v / total, 2) for k, v in out.items() if v > 0}

    def animation_target(self) -> dict[str, float]:
        """Animation share in percent of shots: full screen cards, and every shot carrying an overlay."""
        m = self.shot_mix()
        return {"full_screen_cards": m.get("motion_graphic", 0.0),
                "overlay_shots": round(m.get("footage_overlay", 0.0) + m.get("image_overlay", 0.0), 2)}


class VoiceConfig(Model):
    provider: str = "elevenlabs"
    voice_id: str = ""  # default voice for every TTS provider without its own entry in voice_ids
    # voice ids differ by provider: tts provider name (or type) -> voice id, e.g. {algrow_tts: "..."}
    voice_ids: dict[str, str] = Field(default_factory=dict)
    model: str = "auto"
    stability: float = 0.45
    similarity: float = 0.8


class LicencePolicy(Model):
    """Per brand licence preferences. Flagging only: nothing is blocked except the blocklist."""

    prefer_creative_commons: bool = False  # YouTube: search with the Creative Commons filter first, then normally


class Brand(Model):
    id: str
    name: str = ""
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    style_profile_id: str = "documentary_default"
    music_folder: str = "fixtures/music"
    intro_path: Optional[str] = None
    outro_path: Optional[str] = None
    blocklist: list[str] = Field(default_factory=list)
    niche_notes: str = ""
    language: str = "en"
    licence_policy: LicencePolicy = Field(default_factory=LicencePolicy)


# ---------------------------------------------------------------- edit ops

class EditOpName(str, Enum):
    swap_shot = "swap_shot"
    upload_shot = "upload_shot"
    regenerate_ai = "regenerate_ai"
    set_duration = "set_duration"
    split_shot = "split_shot"
    merge_shots = "merge_shots"
    change_type = "change_type"
    set_overlay = "set_overlay"
    remove_overlay = "remove_overlay"
    reorder = "reorder"
    retime_section = "retime_section"
    change_music = "change_music"
    set_caption_style = "set_caption_style"
    rewrite_beat = "rewrite_beat"


class EditOp(Model):
    """One structured edit. args keys by op:
    swap_shot: shot_id, candidate_id
    upload_shot: shot_id, path
    regenerate_ai: shot_id, prompt_delta
    set_duration: shot_id, seconds
    split_shot: shot_id, at (seconds into shot)
    merge_shots: a, b
    change_type: shot_id, type, optional add_must_avoid, add_queries (lists), reject_selected (reason)
    set_overlay: shot_id, motion_spec
    remove_overlay: shot_id
    reorder: shot_id, new_index
    retime_section: scene_id or range [start_s, end_s], factor
    change_music: track
    set_caption_style: delta (dict of CaptionStyle fields)
    rewrite_beat: beat_id, instruction
    """

    op: EditOpName
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""


class EditResult(Model):
    ops: list[EditOp]
    applied: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dirty_shots: list[str] = Field(default_factory=list)
    needs_renarration: bool = False
    force: list[str] = Field(default_factory=list)  # stages that must rerun after these ops


# ---------------------------------------------------------------- reports

class CostEntry(Model):
    ts: float = Field(default_factory=time.time)
    role: str
    provider: str
    model: str = ""
    stage: str = ""
    units: float = 0.0
    unit_kind: str = ""  # tokens, images, characters, seconds
    cost_usd: float = 0.0
    mock: bool = False
    ok: bool = True
    latency_s: float = 0.0
    note: str = ""
    credits: Optional[float] = None  # subscription credits the provider reported (cost_usd stays 0)


class CostReport(Model):
    project_id: str
    entries: list[CostEntry] = Field(default_factory=list)
    total_usd: float = 0.0
    per_minute_usd: float = 0.0
    mock_calls: int = 0


class QAShot(Model):
    shot_id: str
    index: int
    start: float
    end: float
    type: str
    score: Optional[float]
    reason: str = ""
    attempts: int = 0
    flags: list[str] = Field(default_factory=list)
    post_render_score: Optional[float] = None


class FallbackNote(Model):
    """A part of the video a real provider should have made but heuristics made instead."""

    role: str  # writer, director
    stage: str  # script, shotlist
    scene_id: str  # s01.., or "all" for whole-script context such as the director's script summary
    reason: str = ""


class QAReport(Model):
    project_id: str
    threshold: float
    shots: list[QAShot]
    pct_above_threshold: float
    pct_no_match_accepted: float
    mix_target: dict[str, float]
    mix_achieved: dict[str, float]
    # Animation in percent of shots: full_screen_cards, overlay_shots (shots carrying any overlay) and, in
    # achieved only, inset_overlays (maps and charts moving over the picture).
    animation_target: dict[str, float] = Field(default_factory=dict)
    animation_achieved: dict[str, float] = Field(default_factory=dict)
    pacing: dict[str, dict[str, float]]
    caption_coverage: float
    caption_word_match: float
    attribution_count: int
    mock_used: list[str] = Field(default_factory=list)
    # Sections a configured real provider failed on that shipped from heuristics (sets needs_review).
    heuristic_fallbacks: list[FallbackNote] = Field(default_factory=list)
    nc_nd_assets: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    standard_youtube_assets: list[str] = Field(default_factory=list)  # selected Standard YouTube Licence clips
    # On-screen text against the narration: text_items, timing_error_s (median gap between a text overlay and its
    # anchor word), unanchored (text overlays whose anchor is not spoken near them), verbatim_pct.
    overlay_text: dict[str, float] = Field(default_factory=dict)


ALL_MODELS: dict[str, type[BaseModel]] = {
    "Project": Project,
    "Research": Research,
    "Outline": Outline,
    "Script": Script,
    "Narration": Narration,
    "MotionSpec": MotionSpec,
    "ShotList": ShotList,
    "DirectorPlan": DirectorPlan,
    "StyleProfile": StyleProfile,
    "Brand": Brand,
    "EditOp": EditOp,
    "CostReport": CostReport,
    "QAReport": QAReport,
}
