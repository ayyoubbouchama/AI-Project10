def can_place(box, x, y, z, container, placed_boxes):
    # check container boundaries
    if x + box.l > container.L or y + box.w > container.W or z + box.h > container.H:
        return False

    # check overlap with other boxes
    for pb in placed_boxes:
        if not (
            x + box.l <= pb["x"] or
            x >= pb["x"] + pb["l"] or
            y + box.w <= pb["y"] or
            y >= pb["y"] + pb["w"] or
            z + box.h <= pb["z"] or
            z >= pb["z"] + pb["h"]
        ):
            return False

    return True


def place_boxes(boxes, container):
    positions = []
    x, y, z = 0, 0, 0
    max_height = 0
    max_depth = 0

    for box in boxes:
        # move in x
        if x + box.l > container.L:
            x = 0
            y += max_depth
            max_depth = 0

        # move in y
        if y + box.w > container.W:
            y = 0
            z += max_height
            max_height = 0

        # stop if no space
        if z + box.h > container.H:
            break

        if can_place(box, x, y, z, container, positions):
            positions.append({
                "id": box.id,
                "x": x,
                "y": y,
                "z": z,
                "l": box.l,
                "w": box.w,
                "h": box.h
            })

            x += box.l
            max_depth = max(max_depth, box.w)
            max_height = max(max_height, box.h)

    return positions