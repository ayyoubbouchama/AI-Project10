# ==============================================================================
# Project 10 - 3D Container Loading Optimizer
# Algorithm: Greedy Best-Fit Decreasing (BFD) + Bottom-Left-Fill (BLF)
# ==============================================================================

class Box:
    def __init__(self, id, l, w, h, weight=1):
        self.id = id
        self.l = l
        self.w = w
        self.h = h
        self.weight = weight


class Container:
    def __init__(self, L, W, H):
        self.L = L
        self.W = W
        self.H = H


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

    total_box_volume = sum(b.l * b.w * b.h for b in boxes)
    container_volume = container.L * container.W * container.H
    utilization = (total_box_volume / container_volume) * 100

    return {
        "positions": positions,
        "utilization": utilization
    }


if __name__ == "__main__":
    container = Container(L=10, W=10, H=10)

    boxes = [
        Box(id=0, l=3, w=3, h=3, weight=5),
        Box(id=1, l=4, w=4, h=2, weight=3),
        Box(id=2, l=2, w=5, h=3, weight=2),
        Box(id=3, l=5, w=2, h=2, weight=4),
        Box(id=4, l=3, w=3, h=4, weight=1),
        Box(id=5, l=2, w=2, h=2, weight=2),
        Box(id=6, l=4, w=3, h=2, weight=3),
        Box(id=7, l=1, w=5, h=3, weight=1),
        Box(id=8, l=3, w=2, h=4, weight=2),
        Box(id=9, l=2, w=4, h=3, weight=1),
    ]

    result = solve(boxes, container)

    print(f"Utilization : {result['utilization']:.2f}%")
    print(f"Boxes packed: {len(result['positions'])} / {len(boxes)}")
    print()
    for p in result["positions"]:
        print(f"  Box {p['id']:2d}  pos=({p['x']:2d}, {p['y']:2d}, {p['z']:2d})  "
              f"dims=({p['l']}×{p['w']}×{p['h']})")
