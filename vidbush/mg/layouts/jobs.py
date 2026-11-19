"""Building a LayoutJob from a spec, the narration words and a pool of candidates."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ...models import Candidate, LayoutSpec
from ..house import House, house_style
from .base import FrameProfile, LayoutJob, PictureRef
from .timing import cue_times


def picture_ref(c: Any) -> Optional[PictureRef]:
    """A PictureRef from a Candidate (or a PictureRef, or a local path); None when there is no file."""
    if c is None:
        return None
    if isinstance(c, PictureRef):
        return c if Path(c.path).exists() else None
    if isinstance(c, (str, Path)):
        p = Path(c)
        if not p.exists():
            return None
        media = "video" if p.suffix.lower() in (".mp4", ".mov", ".webm", ".mkv", ".m4v") else "image"
        return PictureRef(str(p), media)
    if isinstance(c, Candidate):
        if not c.local_path or not Path(c.local_path).exists():
            return None
        src = c.source.value if hasattr(c.source, "value") else str(c.source)
        return PictureRef(c.local_path, c.media, c.in_point, src, c.title, c.width, c.height)
    return None


def make_job(spec: LayoutSpec, words: Sequence[Any], clip_start: float, duration: float,
             pool: Mapping[str, Any], frame: Optional[FrameProfile] = None, house: Optional[House] = None,
             width: int = 1920, height: int = 1080, fps: int = 30,
             extra: Optional[dict[str, Any]] = None) -> LayoutJob:
    """Cues from the narration (clip_start is the narration time the clip starts at), pictures from pool
    (keyed by Candidate.id; a picture value that is an existing path is used directly)."""
    cues = cue_times(spec.elements, words, clip_start, clip_start + duration)
    near = [{"text": w["text"] if isinstance(w, dict) else w.text,
             "start": float(w["start"] if isinstance(w, dict) else w.start),
             "end": float(w["end"] if isinstance(w, dict) else w.end)} for w in words]
    near = [w for w in near if clip_start - 2.0 <= w["start"] <= clip_start + duration + 2.0]
    ex = {"words": near, "clip_start": clip_start}  # word by word effects (a quote underlined as it is read)
    ex.update(extra or {})
    job = LayoutJob(spec=spec, cues=cues, duration=duration, width=width, height=height, fps=fps,
                    frame=frame or FrameProfile(), house=house or house_style(), extra=ex)
    for i, el in enumerate(spec.elements):
        if not el.picture:
            continue
        ref = picture_ref(pool.get(el.picture)) or picture_ref(el.picture)
        if ref is None:
            job.missing.append(f"element {i} ({el.role}): picture {el.picture} not found")
        elif ref.generated:
            job.missing.append(f"element {i} ({el.role}): generated picture refused, layouts show real pictures only")
        else:
            job.pictures[i] = ref
    for c in cues:  # a layout is shown whole or not at all: an element it cannot time makes it incomplete
        el = spec.elements[c.index]
        if c.outside:
            job.missing.append(f"element {c.index} ({el.role}): its words are spoken outside the shot")
        elif el.anchor and not c.found:
            job.missing.append(f"element {c.index} ({el.role}): anchor '{el.anchor}' not spoken near the clip")
    return job
