def can_place(box, x, y, z, container, placed_boxes):
    """
    Check if a box can be placed at (x, y, z) without:
    - exceeding container boundaries
    - overlapping any already-placed box
    """
    # Boundary check
    if (x + box.l > container.L + 1e-9 or
        y + box.w > container.W + 1e-9 or
        z + box.h > container.H + 1e-9):
        return False

    # Overlap check
    for pb in placed_boxes:
        if not (
            x + box.l <= pb["x"] + 1e-9 or
            x >= pb["x"] + pb["l"] - 1e-9 or
            y + box.w <= pb["y"] + 1e-9 or
            y >= pb["y"] + pb["w"] - 1e-9 or
            z + box.h <= pb["z"] + 1e-9 or
            z >= pb["z"] + pb["h"] - 1e-9
        ):
            return False

    return True


def is_supported(box, x, y, z, placed_boxes):
    """
    A box is supported if:
    - it rests on the floor (z == 0), OR
    - it rests on top of another placed box.

    This enforces the gravity constraint required by the assignment.
    """
    if z == 0:
        return True

    for pb in placed_boxes:
        # pb's top face is at pb["z"] + pb["h"]
        if abs((pb["z"] + pb["h"]) - z) < 1e-9:
            # Check horizontal overlap (at least partial support)
            x_overlap = (x < pb["x"] + pb["l"]) and (x + box.l > pb["x"])
            y_overlap = (y < pb["y"] + pb["w"]) and (y + box.w > pb["y"])
            if x_overlap and y_overlap:
                return True

    return False


def generate_extreme_points(placed_boxes, container):
    """
    Generate candidate placement points (Extreme Points strategy).

    After each placement, new corners are generated from:
    - The right face (x + l) of every placed box
    - The front face (y + w) of every placed box
    - The top face  (z + h) of every placed box

    These are the only locations where a new box can improve packing density.
    The origin (0, 0, 0) is always included.
    """
    points = {(0, 0, 0)}

    for pb in placed_boxes:
        points.add((pb["x"] + pb["l"], pb["y"], pb["z"]))
        points.add((pb["x"], pb["y"] + pb["w"], pb["z"]))
        points.add((pb["x"], pb["y"], pb["z"] + pb["h"]))

    # Filter out points already outside container
    points = [
        (x, y, z) for x, y, z in points
        if x < container.L and y < container.W and z < container.H
    ]

    # Sort: prefer low z (gravity), then low y, then low x
    points.sort(key=lambda p: (p[2], p[1], p[0]))
    return points


def place_boxes(boxes, container):
    """
    Place boxes into the container using the Extreme Points heuristic.

    For each box (in the given order), try every extreme point and pick
    the first valid position that satisfies:
    - boundary constraints
    - no overlap
    - gravity support

    The order of boxes (their permutation) is what the GA/SA/Greedy controls.
    This function is the shared decoder used by ALL algorithms.
    """
    positions = []

    for box in boxes:
        extreme_points = generate_extreme_points(positions, container)
        placed = False

        for (x, y, z) in extreme_points:
            if can_place(box, x, y, z, container, positions) and \
               is_supported(box, x, y, z, positions):
                positions.append({
                    "id": box.id,
                    "x": x,
                    "y": y,
                    "z": z,
                    "l": box.l,
                    "w": box.w,
                    "h": box.h
                })
                placed = True
                break

        # If no extreme point works, this box doesn't fit — skip it.
        # (The algorithm will be penalized via utilization or fitness score.)

    return positions