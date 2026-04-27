import json
from core.models import Box, Container


def load_data(path):
    with open(path) as f:
        data = json.load(f)

    # Container
    c = data["container"]["inner_dimensions"]
    container = Container(c["length_cm"], c["width_cm"], c["height_cm"])

    # Boxes — expand each box_type by quantity
    boxes = []
    id_counter = 0

    for bt in data["box_types"]:
        for _ in range(bt["quantity"]):
            boxes.append(
                Box(
                    id=id_counter,
                    l=bt["length_cm"],
                    w=bt["width_cm"],
                    h=bt["height_cm"],
                    weight=bt["weight_kg"]
                )
            )
            id_counter += 1

    return boxes, container