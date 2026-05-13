

from core.models import Box, Container
from collections import defaultdict
import math


# ──────────────────────────────────────────────────────────────────────────────
# Rotation helpers
# ──────────────────────────────────────────────────────────────────────────────

_ROTATIONS = [
    (0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)
]

def _get_rotations(box, CL, CW, CH):
    d = [box.l, box.w, box.h]
    locked = getattr(box, 'orientation_locked', False) or getattr(box, 'fragile', False)
    if locked:
        cands = [(d[0],d[1],d[2]), (d[1],d[0],d[2])]
    else:
        seen, cands = set(), []
        for i,j,k in _ROTATIONS:
            t = (d[i],d[j],d[k])
            if t not in seen:
                seen.add(t); cands.append(t)
    return [(l,w,h) for l,w,h in cands
            if l<=CL+1e-9 and w<=CW+1e-9 and h<=CH+1e-9]


# ──────────────────────────────────────────────────────────────────────────────
# Spatial hash grid
# ──────────────────────────────────────────────────────────────────────────────

class SpatialGrid:
    """
    Uniform 3-D grid for fast overlap detection.

    CELL_SIZE is chosen once based on the average box edge so that each
    box spans ~1-2 cells on average — this minimises both the number of
    cells to register per box and the number of candidates to check.
    """

    def __init__(self, CL, CW, CH, cell_size):
        self.CL = CL
        self.CW = CW
        self.CH = CH
        self.cs = cell_size
        self.cells = defaultdict(list)   # (ix,iy,iz) → [box_dict, ...]

    # ── helpers ──────────────────────────────────────────────────────

    def _range(self, lo, hi):
        """Integer cell indices covering interval [lo, hi]."""
        cs = self.cs
        return range(int(lo / cs), int((hi - 1e-9) / cs) + 1)

    def _cells_for(self, x, y, z, l, w, h):
        """All grid cells touched by a box."""
        for ix in self._range(x, x+l):
            for iy in self._range(y, y+w):
                for iz in self._range(z, z+h):
                    yield (ix, iy, iz)

    # ── public API ───────────────────────────────────────────────────

    def register(self, p):
        """Add a placed-box dict to every cell it occupies."""
        for cell in self._cells_for(p["x"], p["y"], p["z"],
                                    p["l"], p["w"], p["h"]):
            self.cells[cell].append(p)

    def candidates(self, x, y, z, l, w, h):
        """Return unique boxes that share at least one cell with (x,y,z,l,w,h)."""
        seen = set()
        result = []
        for cell in self._cells_for(x, y, z, l, w, h):
            for p in self.cells.get(cell, []):
                pid = id(p)
                if pid not in seen:
                    seen.add(pid)
                    result.append(p)
        return result

    def overlaps_any(self, x, y, z, l, w, h, eps=1e-6):
        """True if the candidate box overlaps any registered box."""
        for p in self.candidates(x, y, z, l, w, h):
            if not (x+l <= p["x"]+eps or p["x"]+p["l"] <= x+eps or
                    y+w <= p["y"]+eps or p["y"]+p["w"] <= y+eps or
                    z+h <= p["z"]+eps or p["z"]+p["h"] <= z+eps):
                return True
        return False


def _make_grid(CL, CW, CH, boxes):
    """
    Pick a cell size ≈ average box edge, clamped to [10, 60] cm.
    Then build and return an empty SpatialGrid.
    """
    if boxes:
        avg_edge = sum((b.l+b.w+b.h)/3 for b in boxes) / len(boxes)
    else:
        avg_edge = 40.0
    cell_size = max(10.0, min(60.0, avg_edge))
    return SpatialGrid(CL, CW, CH, cell_size)


# ──────────────────────────────────────────────────────────────────────────────
# Support check  (uses grid — only checks boxes near z-level)
# ──────────────────────────────────────────────────────────────────────────────

def _is_supported(x, y, l, w, z, grid, threshold=0.10, eps=1.0):
    """
    True if ≥ threshold fraction of the box base is supported.
    Uses the spatial grid to find only boxes near z (fast).
    """
    if z < 1e-6:
        return True
    base = l * w
    if base < 1e-9:
        return True

    # Query a thin slab just below z
    nearby = grid.candidates(x, y, z - eps, l, w, eps * 2)
    supported = 0.0
    for p in nearby:
        if abs((p["z"] + p["h"]) - z) < eps:
            ox = max(0.0, min(x+l, p["x"]+p["l"]) - max(x, p["x"]))
            oy = max(0.0, min(y+w, p["y"]+p["w"]) - max(y, p["y"]))
            supported += ox * oy
            if supported / base >= threshold:
                return True
    return False


# ──────────────────────────────────────────────────────────────────────────────
# Corner generation  (fast O(n), same as original)
# ──────────────────────────────────────────────────────────────────────────────

def _corners(placed):
    pts = [(0.0, 0.0, 0.0)]
    for p in placed:
        pts.append((p["x"]+p["l"], p["y"],          p["z"]))
        pts.append((p["x"],        p["y"]+p["w"],   p["z"]))
        pts.append((p["x"],        p["y"],           p["z"]+p["h"]))
    return pts


# ──────────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────────

def _validate(positions, CL, CW, CH, threshold=0.10):
    err = {"overlaps": 0, "boundary": 0, "floating": 0}
    eps = 1e-6

    for p in positions:
        if (p["x"] < -eps or p["y"] < -eps or p["z"] < -eps or
            p["x"]+p["l"] > CL+eps or
            p["y"]+p["w"] > CW+eps or
            p["z"]+p["h"] > CH+eps):
            err["boundary"] += 1

    n = len(positions)
    for i in range(n):
        a = positions[i]
        for j in range(i+1, n):
            b = positions[j]
            if not (a["x"]+a["l"] <= b["x"]+eps or b["x"]+b["l"] <= a["x"]+eps or
                    a["y"]+a["w"] <= b["y"]+eps or b["y"]+b["w"] <= a["y"]+eps or
                    a["z"]+a["h"] <= b["z"]+eps or b["z"]+b["h"] <= a["z"]+eps):
                err["overlaps"] += 1

    grid = _make_grid(CL, CW, CH, [])
    for p in positions:
        grid.register(p)
    for p in positions:
        if p["z"] >= 1e-6:
            nearby = grid.candidates(p["x"], p["y"],
                                     p["z"]-1.0, p["l"], p["w"], 2.0)
            base = p["l"] * p["w"]
            sup  = 0.0
            for q in nearby:
                if q is not p and abs((q["z"]+q["h"]) - p["z"]) < 1.0:
                    ox = max(0.0, min(p["x"]+p["l"], q["x"]+q["l"]) - max(p["x"], q["x"]))
                    oy = max(0.0, min(p["y"]+p["w"], q["y"]+q["w"]) - max(p["y"], q["y"]))
                    sup += ox * oy
            if sup / max(base, 1e-9) < threshold:
                err["floating"] += 1

    err["valid"] = (err["overlaps"] == 0 and
                    err["boundary"] == 0 and
                    err["floating"] == 0)
    return err


# ──────────────────────────────────────────────────────────────────────────────
# Solver
# ──────────────────────────────────────────────────────────────────────────────

class GreedyBFD:
    """
    Greedy Best-Fit Decreasing solver — spatial-grid accelerated.

    For each box (sorted by volume ↓, weight ↓):
      1. Generate candidate corners  O(n)
      2. For each corner × rotation:
           a. Boundary check              O(1)
           b. Gravity check  (grid)       O(1) avg
           c. Overlap check  (grid)       O(1) avg   ← was O(n), now fast
           d. BLF score: min(z, y, x)
      3. Place at best valid corner, register in grid.

    Result: same quality as v2, ~20× faster.
    """

    SUPPORT_THRESHOLD = 0.10

    @staticmethod
    def solve(boxes, container):
        CL, CW, CH = container.L, container.W, container.H
        thresh = GreedyBFD.SUPPORT_THRESHOLD

        sorted_boxes = sorted(
            boxes,
            key=lambda b: (-(b.l*b.w*b.h),
                           -getattr(b,'weight',0),
                           -(b.l*b.w))
        )

        grid      = _make_grid(CL, CW, CH, boxes)
        placed    = []
        positions = []

        for box in sorted_boxes:
            rotations = _get_rotations(box, CL, CW, CH)
            if not rotations:
                continue

            corners  = _corners(placed)
            best_pos = None
            best_rot = None
            best_key = None

            for (cx, cy, cz) in corners:
                for (bl, bw, bh) in rotations:

                    # 1. Boundary
                    if (cx < -1e-9 or cy < -1e-9 or cz < -1e-9 or
                        cx+bl > CL+1e-9 or cy+bw > CW+1e-9 or cz+bh > CH+1e-9):
                        continue

                    # 2. Gravity
                    if not _is_supported(cx, cy, bl, bw, cz,
                                         grid, threshold=thresh):
                        continue

                    # 3. Overlap  ← grid makes this O(1) average
                    if grid.overlaps_any(cx, cy, cz, bl, bw, bh):
                        continue

                    # 4. BLF score
                    key = (round(cz,4), round(cy,4), round(cx,4))
                    if best_key is None or key < best_key:
                        best_key = key
                        best_pos = (cx, cy, cz)
                        best_rot = (bl, bw, bh)

            if best_pos is not None:
                x, y, z    = best_pos
                bl, bw, bh = best_rot
                p = {"x":x, "y":y, "z":z, "l":bl, "w":bw, "h":bh}
                placed.append(p)
                grid.register(p)
                positions.append({"id":box.id,
                                   "x":x,  "y":y,  "z":z,
                                   "l":bl, "w":bw, "h":bh})

        placed_vol  = sum(p["l"]*p["w"]*p["h"] for p in positions)
        utilization = placed_vol / (CL*CW*CH) * 100.0
        validation  = _validate(positions, CL, CW, CH, threshold=thresh)

        return {
            "positions":   positions,
            "utilization": utilization,
            "validation":  validation,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatible shim
# ──────────────────────────────────────────────────────────────────────────────

def solve(boxes, container):
    return GreedyBFD.solve(boxes, container)