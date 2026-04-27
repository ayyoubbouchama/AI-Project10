# ==============================================================================
# Project 10 - 3D Container Loading Optimizer
# Algorithm: Greedy Best-Fit Decreasing (BFD) + Bottom-Left-Fill (BLF)
# ==============================================================================
from core.models import Box, Container
def solve(boxes, container):
    positions = []

    sorted_boxes = sorted(boxes, key=lambda b: b.l * b.w * b.h, reverse=True)

    placed = []

    for box in sorted_boxes:
        candidate_corners = [(0, 0, 0)]

        for p in placed:
            candidate_corners.append((p["x"] + p["l"], p["y"], p["z"]))
            candidate_corners.append((p["x"], p["y"] + p["w"], p["z"]))
            candidate_corners.append((p["x"], p["y"], p["z"] + p["h"]))

        best_position = None
        best_key = None

        for (cx, cy, cz) in candidate_corners:
            if (cx + box.l > container.L or
                cy + box.w > container.W or
                cz + box.h > container.H):
                continue

            overlaps = False
            for p in placed:
                if (cx < p["x"] + p["l"] and cx + box.l > p["x"] and
                    cy < p["y"] + p["w"] and cy + box.w > p["y"] and
                    cz < p["z"] + p["h"] and cz + box.h > p["z"]):
                    overlaps = True
                    break

            if overlaps:
                continue

            key = (cz, cy, cx)
            if best_key is None or key < best_key:
                best_key = key
                best_position = (cx, cy, cz)

        if best_position is not None:
            x, y, z = best_position

            placed.append({
                "x": x, "y": y, "z": z,
                "l": box.l, "w": box.w, "h": box.h
            })

            positions.append({
                "id": box.id,
                "x": x,
                "y": y,
                "z": z,
                "l": box.l,
                "w": box.w,
                "h": box.h
            })

    placed_volume = sum(p["l"] * p["w"] * p["h"] for p in positions)
    container_volume = container.L * container.W * container.H
    utilization = (placed_volume / container_volume) * 100

    return {
        "positions": positions,
        "utilization": utilization
    }


