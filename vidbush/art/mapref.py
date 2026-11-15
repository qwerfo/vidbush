"""The exact map view as a clean reference, its land mask, and the registration check.

Principle: the geography comes from our renderer, the paint from the model. The reference is the
same view the map renderer draws (same projection, center and zoom), with land, water, lakes, major
rivers and country borders only: no graticule, no labels, no text, no grain. The model paints over
it; registration then measures how far the painted coastline is from the true one, and a painting
that drifted beyond tolerance is never used.

View: a painted base stands for the frame at the map's widest zoom (min of zoom and zoom_end). The
map renderer's static base covers exactly that frame (vidbush/mg/mapdraw.MapBase: S times the frame
at S times the pixels), so one painted image serves the full screen map and its inset alike.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np

from ..mg import mapdraw as M
from ..mg.palettes import get_palette
from ..models import MapPayload

REF_LAND = (214, 203, 176)  # warm tan: reads as land to every model, distinct from water
REF_WATER = (92, 140, 190)  # mid blue
REF_BORDER = (120, 104, 84)
REF_RIVER = (92, 140, 190)
MIN_SHARE = 0.01  # a view with less than 1 percent water or land has no coastline to check


def view_key(p: MapPayload) -> dict[str, float]:
    """What a painted base depends on: the center and the widest zoom (rounded so floats compare)."""
    lat, lon = (list(p.center) + [0.0, 0.0])[:2]
    z = min(float(p.zoom), float(p.zoom_end if p.zoom_end is not None else p.zoom))
    z = float(np.clip(z, 0.6, 12))
    z_hi = max(float(p.zoom), float(p.zoom_end if p.zoom_end is not None else p.zoom))
    return {"lat": round(float(np.clip(lat, -85, 85)), 4), "lon": round(float(lon), 4), "zoom": round(z, 4),
            "detail": bool(z_hi >= 3)}  # the coast layer (50 m from zoom 3) the painting was registered to


class _RefBase(M.MapBase):
    """MapBase with a flat, text free reference look; geometry code is the renderer's own."""

    def _render(self) -> np.ndarray:  # noqa: D401
        # the coast layer the renderer will draw: 50 m when the push-in reaches zoom 3, like MapBase
        detail = getattr(self.c, "detail_zoom", self.z_lo) >= 3
        self.countries = M.load_layer("ne_50m_admin_0_countries.geojson" if detail else "ne_110m_admin_0_countries.geojson")
        k = max(1.0, self.k)
        img = np.empty((self.bh, self.bw, 3), np.uint8)
        img[...] = REF_WATER
        land = self.rings(self.countries)
        if land:
            cv2.fillPoly(img, [np.round(r * 16).astype(np.int32) for r in land], REF_LAND, cv2.LINE_AA, 4)
        for r in self._lakes():
            cv2.fillPoly(img, [np.round(r * 16).astype(np.int32)], REF_WATER, cv2.LINE_AA, 4)
        for r in self._rivers():
            M.polyline(img, r, REF_RIVER, max(1.0, 2.4 * k))
        for r in land:
            M.polyline(img, r, REF_BORDER, max(1.0, 1.4 * k))
        return img

    def _lakes(self) -> list[np.ndarray]:
        z = self.z_lo
        return self.rings(M.load_layer("ne_50m_lakes.geojson"),
                          None if z >= 3.5 else (lambda p: (p.get("scalerank") or 9) <= 1))

    def _rivers(self) -> list[np.ndarray]:
        if self.z_lo < 3.5:
            return []
        layer = [f for f in M.load_layer(M.RIVERS) if (f[0].get("scalerank") or 9) <= M.RIVER_RANK]
        return self.rings(layer)

    def inland_water(self) -> np.ndarray:
        """True lakes and major rivers (255) at the base size: the only places painted inland water may be."""
        m = np.zeros((self.bh, self.bw), np.uint8)
        lakes = self._lakes()
        if lakes:
            cv2.fillPoly(m, [np.round(r * 16).astype(np.int32) for r in lakes], 255, cv2.LINE_AA, 4)
        for r in self._rivers():
            cv2.polylines(m, [np.round(r * 16).astype(np.int32)], False, 255, max(1, int(3 * self.k)), cv2.LINE_AA, 4)
        return m

    def mask(self, lakes: bool = False) -> np.ndarray:
        """Land 255, sea 0 (lakes 0 too when asked), at the base size."""
        m = np.zeros((self.bh, self.bw), np.uint8)
        land = self.rings(self.countries)
        if land:
            cv2.fillPoly(m, [np.round(r * 16).astype(np.int32) for r in land], 255, cv2.LINE_AA, 4)
        rings = self._lakes() if lakes else []
        if rings:
            cv2.fillPoly(m, [np.round(r * 16).astype(np.int32) for r in rings], 0, cv2.LINE_AA, 4)
        return m


def _base(p: MapPayload, width: int, height: int, zoom: float) -> _RefBase:
    v = view_key(p)
    proj = M.Projection(v["lat"], v["lon"], width, height)
    z_hi = max(float(p.zoom), float(p.zoom_end if p.zoom_end is not None else p.zoom))
    comp = SimpleNamespace(W=width, H=height, proj=proj, pal=get_palette("documentary_warm"),
                           k=min(width / 1920.0, height / 1080.0), C=3, detail_zoom=z_hi,
                           p=SimpleNamespace(shaded_regions=[]))
    # zoom >= 3 picks the 50 m countries like the renderer does for a detailed push-in
    return _RefBase(comp, zoom, zoom)  # type: ignore[arg-type]


def pad_zoom(zoom: float, pad: float) -> float:
    """Zoom whose frame is the view widened so that trimming `pad` from every edge gives the view back."""
    if pad <= 0:
        return zoom
    return zoom - math.log2(1.0 / (1.0 - 2.0 * pad))


def reference(p: MapPayload, width: int = 1536, height: int = 864, pad: float = 0.0) -> np.ndarray:
    """RGB reference of the view (widened by pad on every edge for providers whose edges get trimmed)."""
    z = view_key(p)["zoom"]
    return _base(p, width, height, pad_zoom(z, pad)).plain


def land_mask(p: MapPayload, width: int = 768, height: int = 432) -> np.ndarray:
    """Land mask of the view itself (no pad). Lakes count as land: registration checks the sea coast,
    the outline a viewer reads as the place; lakes and rivers are inland detail the painting follows
    from the reference and the renderer labels."""
    return _base(p, width, height, view_key(p)["zoom"]).mask()


def inland_water(p: MapPayload, width: int = 768, height: int = 432) -> np.ndarray:
    return _base(p, width, height, view_key(p)["zoom"]).inland_water()


PAINT_SHARE = 0.12  # a view is painted only when sea and land each cover at least this share of it


def paintable(mask: np.ndarray) -> bool:
    """Enough coastline to hold a painting to the view: on views that were almost all land, every model painted
    an oblique scene (perspective mountains, a train) instead of the map, and registration rightly failed them."""
    share = float((mask > 127).mean())
    return PAINT_SHARE <= share <= 1.0 - PAINT_SHARE


def checkable(mask: np.ndarray) -> bool:
    """A coastline can be checked only when the view holds both land and water."""
    share = float((mask > 127).mean())
    return MIN_SHARE <= share <= 1.0 - MIN_SHARE


# ---------------------------------------------------------------- registration

def _lab(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.GaussianBlur(img, (0, 0), 1.2), cv2.COLOR_RGB2LAB).astype(np.float32)


def classify(painted: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    """Land mask of a painting, learnt from the painting itself.

    Colour centers are fitted to the painting's own pixels deep inside the reference land and deep
    inside the reference water (well away from the coast, where a small drift cannot reach), several
    per class because land is forest, snow and rock at once. Every pixel takes the class of its
    nearest center. A painting that moved the coast puts water colours where the reference has
    land (or the reverse), which the boundary measure then sees.
    """
    h, w = ref_mask.shape
    img = cv2.resize(painted, (w, h), interpolation=cv2.INTER_AREA)
    lab = _lab(img).reshape(-1, 3)
    r = max(3, int(round(0.03 * w)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    deep_land = cv2.erode((ref_mask > 127).astype(np.uint8), ker).reshape(-1) > 0
    deep_water = cv2.erode((ref_mask <= 127).astype(np.uint8), ker).reshape(-1) > 0
    centers, classes = [], []
    rng = np.random.default_rng(7)
    for sel, cls, k in ((deep_land, 1, 6), (deep_water, 0, 4)):
        pts = lab[sel]
        if len(pts) < 50:
            continue
        if len(pts) > 20000:
            pts = pts[rng.choice(len(pts), 20000, replace=False)]
        k = min(k, max(1, len(pts) // 50))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        cv2.setRNGSeed(7)
        _, _, c = cv2.kmeans(pts.astype(np.float32), k, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        centers.append(c)
        classes += [cls] * len(c)
    if not centers:
        return np.zeros_like(ref_mask)
    cen = np.vstack(centers)
    d = ((lab[:, None, :] - cen[None, :, :]) ** 2).sum(-1)
    out = (np.asarray(classes)[d.argmin(1)] * 255).astype(np.uint8).reshape(h, w)
    out = cv2.medianBlur(out, 5)
    return _drop_specks(out, max(4, int(0.0005 * h * w)))


def _drop_specks(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove land and water islands smaller than min_area pixels (texture noise, not geography)."""
    out = mask.copy()
    for val, fill in ((255, 0), (0, 255)):
        n, lab, stats, _ = cv2.connectedComponentsWithStats((out == val).astype(np.uint8), 8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < min_area:
                out[lab == i] = fill
    return out


def _edge(mask: np.ndarray) -> np.ndarray:
    e = cv2.morphologyEx((mask > 127).astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    e[:2, :] = e[-2:, :] = False
    e[:, :2] = e[:, -2:] = False  # the frame edge is not a coastline
    return e


MAX_SCALE_DRIFT = 0.06  # a painting may sit up to 6 percent larger or smaller (canvas fit), corrected
MAX_SHIFT = 0.04  # and up to 4 percent of the frame off center, corrected


def align(pred: np.ndarray, ref: np.ndarray) -> Optional[np.ndarray]:
    """2x3 affine (scale and shift per axis, no rotation) mapping the painting onto the reference, or
    None when no small correction fits. Models fit a reference to their own canvas with a slight
    stretch or offset; that is corrected (the painted base is warped with the same matrix), anything
    larger is not a canvas fit and fails registration."""
    a = cv2.GaussianBlur(ref.astype(np.float32) / 255.0, (0, 0), 3)
    b = cv2.GaussianBlur(pred.astype(np.float32) / 255.0, (0, 0), 3)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(a, b, warp, cv2.MOTION_AFFINE,
                                       (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-5), None, 5)
    except cv2.error:
        return np.eye(2, 3, dtype=np.float32)
    h, w = ref.shape
    sx, sy, shx, shy = warp[0, 0], warp[1, 1], warp[0, 1], warp[1, 0]
    if (abs(sx - 1) > MAX_SCALE_DRIFT or abs(sy - 1) > MAX_SCALE_DRIFT or abs(shx) > 0.02 or abs(shy) > 0.02
            or abs(warp[0, 2]) > MAX_SHIFT * w or abs(warp[1, 2]) > MAX_SHIFT * h):
        return None
    # keep only scale and shift: rotation or shear would mean the model redrew the land, not refit it
    warp[0, 1] = warp[1, 0] = 0.0
    return warp


def warp_to_ref(img: np.ndarray, warp: np.ndarray, size: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """Apply align()'s matrix (reference coordinates to painting coordinates) at any image size."""
    w, h = size
    ih, iw = img.shape[:2]
    rh, rw = REG_SIZE[1], REG_SIZE[0]
    # the matrix was found at REG_SIZE: rescale it to (iw, ih) -> (w, h)
    S_out = np.diag([rw / w, rh / h, 1.0])
    S_in = np.diag([iw / rw, ih / rh, 1.0])
    m = S_in @ np.vstack([warp, [0, 0, 1]]) @ S_out
    flags = (cv2.INTER_NEAREST if nearest else cv2.INTER_LANCZOS4) | cv2.WARP_INVERSE_MAP
    return cv2.warpAffine(img, m[:2].astype(np.float32), (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT)


REG_SIZE = (768, 432)


def register(painted: np.ndarray, p: MapPayload) -> dict[str, Any]:
    """Registration of a painting (RGB, any size, covering the view) against the view's true land.

    Returns {checkable, warp, iou, drift_p90, invent_p90, worst_patch, ok, reason}. drift_p90 is the
    90th percentile distance from a true coast pixel to the painted coast, invent_p90 the same from a
    painted coast pixel to the true coast (both information, percent of the frame width); worst_patch
    and iou decide (judge_metrics, TOLERANCE).
    """
    ref = land_mask(p, *REG_SIZE)
    out: dict[str, Any] = {"checkable": checkable(ref), "warp": None, "iou": 0.0, "drift_p90": 99.0,
                           "invent_p90": 99.0, "worst_patch": 99.0, "ok": False, "reason": ""}
    if not out["checkable"]:
        out["reason"] = "view has no coastline to check"
        return out
    pred = classify(painted, ref)
    warp = align(pred, ref)
    if warp is None:
        out["reason"] = "painting is not a small refit of the view (scale, shift or rotation too large)"
        return out
    pred_w = warp_to_ref(pred, warp, REG_SIZE, nearest=True)
    out["warp"] = warp.tolist()
    # compare at the same level of detail: the painting's mask lost islets and inlets to smoothing
    h, w = ref.shape
    ref_s = _drop_specks(cv2.medianBlur(ref, 5), max(4, int(0.0005 * h * w)))
    out.update(measure(inland_as_land(pred_w, ref_s, inland_water(p, *REG_SIZE)), ref_s))
    out["ok"], out["reason"] = judge_metrics(out)
    return out


INLAND = 0.03  # water this far (share of frame width) inside the true land is a lake or river, not a coast
NEAR_WATER = 0.02  # painted inland water within this share of the width of a true lake or river is that water


def inland_as_land(pred: np.ndarray, ref: np.ndarray, allowed: Optional[np.ndarray] = None) -> np.ndarray:
    """Painted water deep inside the true land counts as land where a true lake or river is near: lakes and
    rivers are inland detail the painting follows loosely from the reference, not the sea coast. Painted water
    far from any true one stays water, so an invented lake or sea inlet is a wrong patch (small ones are thinner
    than THICK and do not count). A moved coast still shows within INLAND of the true one; land painted deep in
    the sea is kept (an invented island is wrong geography)."""
    w = ref.shape[1]
    dist = cv2.distanceTransform((ref > 127).astype(np.uint8), cv2.DIST_L2, 3)
    deep = dist > INLAND * w
    if allowed is not None:
        r = max(1, int(round(NEAR_WATER * w)))
        near = cv2.dilate((allowed > 127).astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
        deep &= near
    out = pred.copy()
    out[deep] = 255
    return out


THICK = 0.012  # a disagreement must be thicker than 1.2 percent of the frame width to count


def measure(pred: np.ndarray, ref: np.ndarray) -> dict[str, float]:
    """Land overlap, coast distances (information) and the worst wrong patch (the decision).

    worst_patch is the largest region where painting and truth disagree after every disagreement
    thinner than THICK is removed, as the diameter of a disc of the same area in percent of the frame
    width. Thin differences are fine coastal detail (islets, fjords) and small refits that no viewer
    reads as wrong geography; a thick one is a coast that moved, a peninsula that grew or a bay that
    vanished, wherever it is, which a percentile over the whole coast would average away.
    """
    a, b = pred > 127, ref > 127
    union = float((a | b).sum()) or 1.0
    iou = float((a & b).sum()) / union
    h, w = ref.shape
    er, ep = _edge(ref), _edge(pred)
    dt_ref = cv2.distanceTransform((~er).astype(np.uint8), cv2.DIST_L2, 3)
    dt_pred = cv2.distanceTransform((~ep).astype(np.uint8), cv2.DIST_L2, 3)
    drift = float(np.percentile(dt_pred[er], 90)) if er.any() else 0.0
    invent = float(np.percentile(dt_ref[ep], 90)) if ep.any() else (99.0 if er.any() else 0.0)
    wrong = (a ^ b).astype(np.uint8)
    r = max(1, int(round(THICK * w / 2)))
    wrong = cv2.morphologyEx(wrong, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(wrong, 8)
    worst = max((int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)), default=0)
    return {"iou": round(iou, 4), "drift_p90": round(100 * drift / w, 3), "invent_p90": round(100 * invent / w, 3),
            "worst_patch": round(100 * 2 * math.sqrt(worst / math.pi) / w, 3)}


# Calibrated on the reference itself, synthetic edits and real paintings (docs/DECISIONS.md, Run 3: art
# layer). TOLERANCE["worst_patch"] is the diameter, in percent of the frame width, of the largest wrong
# region a painting may have.
TOLERANCE = {"worst_patch": 4.0, "iou": 0.9}


def judge_metrics(m: dict[str, Any]) -> tuple[bool, str]:
    bad = []
    if m["worst_patch"] > TOLERANCE["worst_patch"]:
        bad.append(f"a wrong land or water patch {m['worst_patch']:.1f}% of the frame width across")
    if m["iou"] < TOLERANCE["iou"]:
        bad.append(f"land overlap {m['iou']:.2f}")
    return (not bad), "; ".join(bad)
