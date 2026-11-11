"""Real route geometry between two places over bundled Natural Earth lines (offline, no routing service).

A ground route (rail, road, river) follows the bundled layer: each endpoint snaps to the nearest line within
SNAP_KM and a heapq Dijkstra over the layer graph gives the path. A route that cannot snap, or whose two
snaps are not connected, has no geometry and is never drawn. Sea and air are a great circle arc, drawn
dashed because it claims no ground path. A mode nobody names ('line') is never a straight line over the land:
it is the real rail line, then the road, else nothing (modes). Trails have no bundled data, so they are never
drawn.
"""
from __future__ import annotations

import heapq
import math
import re
from typing import Optional

import numpy as np

LAYERS = {"rail": "ne_10m_railroads.geojson", "road": "ne_10m_roads.geojson",
          "river": "ne_50m_rivers_lake_centerlines.geojson"}
ARC_VIAS = {"sea", "air"}
VIAS = ("road", "rail", "river", "sea", "trail", "air", "line")
SNAP_KM = 10.0  # an endpoint farther than this from the line is not on it
BRIDGE_KM = 1.0  # a dangling line end this close to another line joins it (junctions lost to simplification)
EARTH_MI = 3958.8
MAX_POINTS = 300  # drawn path points kept in the payload

LatLon = tuple[float, float]

# Words that name how a journey went, checked in this order (a train ride to a river town is still rail).
_VIA_WORDS = [
    ("rail", r"rail\w*|train|trains|locomotive|tracks?|depot|flag stop"),
    ("road", r"road|roads|highway|highways|drive|drove|driving|truck\w*|bus|car|cars|motorway|interstate"),
    ("river", r"river|rivers|upriver|downriver|upstream|downstream|barge\w*|riverboat\w*|paddle\w*"),
    ("air", r"fly|flew|flown|flight\w*|plane\w*|aircraft|airlift\w*|pilot\w*|airmail|air"),
    ("sea", r"sea|seas|ocean|sail\w*|ship\w*|ferry|ferries|boat\w*|voyage\w*|steam\w*|tanker\w*|icebreaker\w*"),
    ("trail", r"trail\w*|sled\w*|dogsled\w*|mush\w*|hik\w*|walk\w*|trek\w*|on foot|snowshoe\w*"),
]
_VIA_RES = [(via, re.compile(rf"\b(?:{words})\b", re.I)) for via, words in _VIA_WORDS]


def infer_via(*texts: str) -> str:
    """How a route travels, from the first text that names a mode (label first, then title or intent).
    'line' when nothing says: an unknown mode, traced as rail, then road (modes)."""
    for text in texts:
        for via, rx in _VIA_RES:
            if text and rx.search(text):
                return via
    return "line"


# ---------------------------------------------------------------- distances

def miles(a: LatLon, b: LatLon) -> float:
    """Great circle distance in statute miles."""
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * EARTH_MI * math.asin(min(1.0, math.sqrt(h)))


def path_miles(points: list[list[float]] | list[LatLon]) -> float:
    """Length of a [lat, lon] polyline in miles."""
    return sum(miles(tuple(points[i]), tuple(points[i + 1])) for i in range(len(points) - 1))


def great_circle(a: LatLon, b: LatLon, n: int = 48) -> list[list[float]]:
    """n points along the great circle from a to b."""
    def vec(p: LatLon) -> np.ndarray:
        la, lo = math.radians(p[0]), math.radians(p[1])
        return np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])
    va, vb = vec(a), vec(b)
    w = math.acos(float(np.clip(va @ vb, -1.0, 1.0)))
    if w < 1e-9:
        return [list(a), list(b)]
    out = []
    for t in np.linspace(0.0, 1.0, n):
        v = (math.sin((1 - t) * w) * va + math.sin(t * w) * vb) / math.sin(w)
        out.append([round(math.degrees(math.asin(v[2])), 4), round(math.degrees(math.atan2(v[1], v[0])), 4)])
    return out


# ---------------------------------------------------------------- graph routing

def _km_xy(lat: np.ndarray, lon: np.ndarray, lat0: float) -> np.ndarray:
    """Local equirectangular kilometres (good to a few percent over a regional bbox)."""
    return np.stack([lon * 111.32 * math.cos(math.radians(lat0)), lat * 110.57], axis=-1)


class _Graph:
    """Nodes are [lat, lon] vertices of the layer lines inside a bbox; edges join consecutive vertices."""

    def __init__(self, layer: str, box: tuple[float, float, float, float]) -> None:
        from .mapdraw import load_layer
        lat_lo, lon_lo, lat_hi, lon_hi = box
        self.lat0 = (lat_lo + lat_hi) / 2
        self.ids: dict[tuple[float, float], int] = {}
        self.pts: list[tuple[float, float]] = []
        self.adj: list[dict[int, float]] = []
        segs: list[tuple[int, int]] = []
        for _props, rings in load_layer(LAYERS[layer]):
            for r in rings:  # r: (n, 2) lon, lat
                if r[:, 1].max() < lat_lo or r[:, 1].min() > lat_hi or r[:, 0].max() < lon_lo or r[:, 0].min() > lon_hi:
                    continue
                prev = None
                for lon, lat in r:
                    k = self._node(float(lat), float(lon))
                    if prev is not None and prev != k:
                        self._edge(prev, k)
                        segs.append((prev, k))
                    prev = k
        self.segs = np.array(segs, np.int64).reshape(-1, 2)
        self.xy = _km_xy(np.array([p[0] for p in self.pts]), np.array([p[1] for p in self.pts]), self.lat0) \
            if self.pts else np.zeros((0, 2))
        self._bridge()

    def _node(self, lat: float, lon: float) -> int:
        key = (round(lat, 3), round(lon, 3))
        k = self.ids.get(key)
        if k is None:
            k = self.ids[key] = len(self.pts)
            self.pts.append(key)
            self.adj.append({})
        return k

    def _edge(self, a: int, b: int, km: Optional[float] = None) -> None:
        w = km if km is not None else miles(self.pts[a], self.pts[b]) * 1.609344
        self.adj[a][b] = self.adj[b][a] = min(w, self.adj[a].get(b, w))

    def _bridge(self) -> None:
        """Join each dangling line end to the nearest other vertex within BRIDGE_KM."""
        cells: dict[tuple[int, int], list[int]] = {}
        for i, (x, y) in enumerate(self.xy):
            cells.setdefault((int(x // BRIDGE_KM), int(y // BRIDGE_KM)), []).append(i)
        for k in [i for i, nb in enumerate(self.adj) if len(nb) == 1]:
            cx, cy = int(self.xy[k][0] // BRIDGE_KM), int(self.xy[k][1] // BRIDGE_KM)
            near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in cells.get((cx + dx, cy + dy), [])
                    if j != k and j not in self.adj[k]]
            if near:
                d = np.hypot(*(self.xy[near] - self.xy[k]).T)
                j = int(np.argmin(d))
                if d[j] <= BRIDGE_KM:
                    self._edge(k, near[j], float(d[j]))

    def snap(self, p: LatLon) -> Optional[int]:
        """New node at the nearest point of any segment to p, joined to that segment's ends; None beyond SNAP_KM."""
        if not len(self.segs):
            return None
        q = _km_xy(np.array(p[0]), np.array(p[1]), self.lat0)
        a, b = self.xy[self.segs[:, 0]], self.xy[self.segs[:, 1]]
        ab = b - a
        t = np.clip(((q - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0.0, 1.0)
        near = a + ab * t[:, None]
        d = np.hypot(*(near - q).T)
        i = int(np.argmin(d))
        if d[i] > SNAP_KM:
            return None
        ia, ib = (int(x) for x in self.segs[i])
        seg_km = float(np.hypot(*ab[i]))
        lat0, lon0 = self.pts[ia]
        lat1, lon1 = self.pts[ib]
        k = len(self.pts)
        self.pts.append((lat0 + (lat1 - lat0) * float(t[i]), lon0 + (lon1 - lon0) * float(t[i])))
        self.adj.append({})
        self._edge(k, ia, seg_km * float(t[i]))
        self._edge(k, ib, seg_km * (1 - float(t[i])))
        return k

    def path(self, a: int, b: int) -> Optional[list[list[float]]]:
        """Shortest path from node a to node b as [lat, lon] points, None when they are not connected."""
        dist, prev, heap = {a: 0.0}, {}, [(0.0, a)]
        while heap:
            d, u = heapq.heappop(heap)
            if u == b:
                break
            if d > dist[u]:
                continue
            for v, w in self.adj[u].items():
                nd = d + w
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        if b not in dist:
            return None
        out, u = [], b
        while u != a:
            out.append(u)
            u = prev[u]
        out.append(a)
        return [[round(self.pts[i][0], 4), round(self.pts[i][1], 4)] for i in reversed(out)]


def _bbox(a: LatLon, b: LatLon) -> tuple[float, float, float, float]:
    """Endpoints' bbox padded by half its size and at least 1 degree, so a path may bend around a bay."""
    pad_lat = max(1.0, abs(a[0] - b[0]) * 0.5)
    pad_lon = max(1.0, abs(a[1] - b[1]) * 0.5)
    return (min(a[0], b[0]) - pad_lat, min(a[1], b[1]) - pad_lon, max(a[0], b[0]) + pad_lat,
            max(a[1], b[1]) + pad_lon)


def ground_path(a: LatLon, b: LatLon, layer: str) -> Optional[list[list[float]]]:
    """Path along a bundled line layer between two points, or None when either end does not snap or the
    two snaps are not connected."""
    # shortcut: the bbox does not cross the antimeridian; a route that does gets no ground path.
    if abs(a[1] - b[1]) > 180:
        return None
    g = _Graph(layer, _bbox(a, b))
    sa, sb = g.snap(a), g.snap(b)
    if sa is None or sb is None or sa == sb:
        return None
    return g.path(sa, sb)


def modes(via: str) -> tuple[str, ...]:
    """Modes a route tries in order: 'line' (nobody named one) is rail, then road; any other is itself."""
    return ("rail", "road") if via == "line" else (via,)


def route_points(a: LatLon, b: LatLon, via: str) -> Optional[list[list[float]]]:
    """Drawn geometry from a to b: the real line for rail, road and river, a great circle for sea and air,
    None for a trail, 'line' (see modes) or a ground route with no line between the two places."""
    if via in LAYERS:
        return ground_path(a, b, via)
    if via in ARC_VIAS:
        return great_circle(a, b)
    return None


def thin(points: list[list[float]], limit: int = MAX_POINTS) -> list[list[float]]:
    """At most about limit points, both ends kept (the drawn line stays on the real one at map scale)."""
    if len(points) <= limit:
        return points
    step = math.ceil(len(points) / limit)
    return points[::step] + ([points[-1]] if (len(points) - 1) % step else [])
