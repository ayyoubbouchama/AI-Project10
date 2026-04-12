def compute_utilization(boxes, container):
    total_box_volume = sum(b.l * b.w * b.h for b in boxes)
    container_volume = container.L * container.W * container.H
    return (total_box_volume / container_volume) * 100