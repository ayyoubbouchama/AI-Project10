def compute_utilization(placed_positions, container):
    """
    Compute the volume utilization of the container.

    Args:
        placed_positions: list of dicts returned by place_boxes()
                          (only boxes that actually fit are included)
        container: Container object with L, W, H

    Returns:
        utilization as a percentage (0.0 - 100.0)

    NOTE: We use placed_positions (not the full box list) so that
    boxes that didn't fit are NOT counted, giving an accurate number.
    """
    if not placed_positions:
        return 0.0

    total_placed_volume = sum(
        p["l"] * p["w"] * p["h"] for p in placed_positions
    )
    container_volume = container.L * container.W * container.H
    return (total_placed_volume / container_volume) * 100


def compute_stability(placed_positions, container):
    """
    Compute a stability score (0.0 - 100.0) measuring how well
    heavy items are placed at the bottom of the container.

    Method:
        For each placed box, compute  weight_i * z_i  (weight × height).
        A perfectly stable packing has all heavy boxes at z=0.
        The worst case has all heavy boxes at the top (z = H - h).

        stability = (1 - actual_sum / worst_case_sum) * 100

    A score of 100 means all heavy boxes are at z=0.
    A score of 0 means the heaviest boxes are as high as possible.
    """
    if not placed_positions:
        return 100.0

    # actual weighted height sum
    actual_sum = 0.0
    worst_sum = 0.0

    for p in placed_positions:
        w = p.get("weight", 1)
        # z is the bottom of the box
        actual_sum += w * p["z"]
        # worst case: this box is as high as possible
        worst_sum += w * max(container.H - p["h"], 0)

    if worst_sum == 0:
        return 100.0

    stability = (1.0 - actual_sum / worst_sum) * 100.0
    return max(0.0, min(100.0, stability))