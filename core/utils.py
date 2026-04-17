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