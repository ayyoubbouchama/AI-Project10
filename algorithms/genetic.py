import random
from core.space_manager import can_place

# -------------------------------
# initialize population (better diversity)
# -------------------------------
def initialize_population(boxes, size=30):
    population = []

    for _ in range(size):
        individual = boxes[:]
        random.shuffle(individual)

        # small variation: sometimes sort partially
        if random.random() < 0.3:
            individual = sorted(individual, key=lambda b: b.l * b.w * b.h, reverse=True)

        population.append(individual)

    return population


# -------------------------------
# fitness function (with penalties)
# -------------------------------
def fitness(solution, container):
    placed = []
    x, y, z = 0, 0, 0
    max_height = 0
    max_depth = 0

    used_volume = 0
    penalty = 0

    for box in solution:

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

        # if exceeds container → penalty
        if z + box.h > container.H:
            penalty += box.l * box.w * box.h
            continue

        # check overlap
        if can_place(box, x, y, z, container, placed):
            placed.append({
                "id": box.id,
                "x": x,
                "y": y,
                "z": z,
                "l": box.l,
                "w": box.w,
                "h": box.h
            })

            used_volume += box.l * box.w * box.h

            x += box.l
            max_depth = max(max_depth, box.w)
            max_height = max(max_height, box.h)
        else:
            # overlap penalty
            penalty += box.l * box.w * box.h

    container_volume = container.L * container.W * container.H

    score = (used_volume - penalty) / container_volume

    return max(score, 0)  # avoid negative fitness


# -------------------------------
# selection (stronger tournament)
# -------------------------------
def selection(population, container, k=3):
    # pick k random individuals
    candidates = random.sample(population, k)

    # return the best among them
    best = candidates[0]
    best_score = fitness(best, container)

    for c in candidates[1:]:
        score = fitness(c, container)
        if score > best_score:
            best = c
            best_score = score

    return best