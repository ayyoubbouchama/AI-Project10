"""
Genetic Algorithm for 3D Container Loading (Bin Packing)
=========================================================

Key design decisions (explained for the report):
-------------------------------------------------
CHROMOSOME:
    A list of (box_index, rotation_index) tuples.
    The ORDER determines how the placer fills the container.
    The rotation_index (0–5) selects which of the 6 axis-aligned
    orientations to use for that box.
    The GA evolves both orderings AND orientations.

FITNESS:
    fitness = utilization(%) - UNPLACED_PENALTY * unplaced_count
              + STABILITY_WEIGHT * stability_score
    We reward high utilization, penalize unplaced boxes, and add a
    small bonus for placing heavy boxes at the bottom.

SELECTION:
    Tournament selection: pick k random individuals, keep the best.
    More robust than roulette wheel for noisy fitness landscapes.

CROSSOVER:
    Order Crossover (OX1): preserves relative ordering of elements.
    Required for permutation chromosomes (classic TSP/bin packing choice).
    Rotation indices are inherited alongside their box index.

MUTATION:
    Three operators applied with independent probabilities:
    - Swap mutation:      swap two random positions.
    - Inversion mutation: reverse a random sub-sequence.
    - Rotation mutation:  change the rotation of a random box.
    Adaptive rates increase during stagnation and decay on improvement.

ELITISM:
    The best individuals always survive to the next generation.
    Prevents accidental loss of the best solution found.

GREEDY SEEDING:
    A few individuals in gen-0 use the greedy sort order, giving
    the GA a strong starting point and faster early convergence.

STAGNATION IMMIGRATION:
    When the best fitness hasn't improved for STAGNATION_LIMIT
    generations, the bottom 20% of the population is replaced with
    fresh random individuals to inject diversity.

EARLY STOPPING:
    If no improvement for EARLY_STOP_GENS generations, terminate
    early to save computation time (important for the web API).
"""

import random
import copy
from core.models import Box
from core.space_manager import place_boxes
from core.utils import compute_utilization, compute_stability


# ─────────────────────────── constants ────────────────────────────

POPULATION_SIZE    = 80       # number of individuals per generation
NUM_GENERATIONS    = 300      # how many generations to run
TOURNAMENT_SIZE    = 5        # candidates per tournament
CROSSOVER_RATE     = 0.90     # probability of crossover vs. direct copy
MUTATION_SWAP_BASE = 0.20     # base probability of swap mutation
MUTATION_INV_BASE  = 0.15     # base probability of inversion mutation
MUTATION_ROT       = 0.20     # probability of rotation mutation per individual
UNPLACED_PENALTY   = 5.0      # % penalty per unplaced box
STABILITY_WEIGHT   = 0.10     # weight for stability bonus in fitness
ELITE_COUNT        = 3        # top N individuals copied unchanged each gen
STAGNATION_LIMIT   = 30       # gens without improvement before immigration
EARLY_STOP_GENS    = 50       # gens without improvement to trigger early stop
IMMIGRATION_RATE   = 0.20     # fraction of population replaced on stagnation


# ─────────────────────── box rotations ────────────────────────────

# 6 axis-aligned orientations: all permutations of (l, w, h)
ROTATIONS = [
    lambda l, w, h: (l, w, h),
    lambda l, w, h: (l, h, w),
    lambda l, w, h: (w, l, h),
    lambda l, w, h: (w, h, l),
    lambda l, w, h: (h, l, w),
    lambda l, w, h: (h, w, l),
]


def rotate_box(box, rotation_index):
    """Return a new Box with dimensions permuted according to rotation_index."""
    rl, rw, rh = ROTATIONS[rotation_index % 6](box.l, box.w, box.h)
    return Box(id=box.id, l=rl, w=rw, h=rh, weight=box.weight)


# ─────────────────────────── chromosome ───────────────────────────

def random_chromosome(n):
    """
    A chromosome is a list of (box_index, rotation_index) tuples.
    The ordering is a random permutation; each box gets a random rotation.
    """
    indices = list(range(n))
    random.shuffle(indices)
    return [(idx, random.randint(0, 5)) for idx in indices]


def greedy_seeded_chromosome(n, boxes):
    """
    Seed a chromosome using the greedy volume-descending order.
    This gives the GA a strong starting point.
    """
    sorted_indices = sorted(range(n), key=lambda i: boxes[i].volume(), reverse=True)
    return [(idx, 0) for idx in sorted_indices]


# ─────────────────────────── fitness ──────────────────────────────

def fitness(chromosome, boxes, container):
    """
    Decode the chromosome into a packing, then score it.

    Score = utilization(%) - UNPLACED_PENALTY * #unplaced_boxes
            + STABILITY_WEIGHT * stability_score

    Higher is better. Maximum possible ≈ 110.0 (perfect fit + perfect stability).
    """
    ordered_boxes = [rotate_box(boxes[idx], rot) for idx, rot in chromosome]
    positions = place_boxes(ordered_boxes, container)

    utilization = compute_utilization(positions, container)
    stability = compute_stability(positions, container)
    unplaced = len(boxes) - len(positions)
    score = utilization - UNPLACED_PENALTY * unplaced + STABILITY_WEIGHT * stability

    return score, positions, utilization


# ─────────────────────────── selection ────────────────────────────

def tournament_select(population, scores, k=TOURNAMENT_SIZE):
    """
    Pick k individuals at random, return the one with the highest score.
    """
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda i: scores[i])
    return [gene for gene in population[best]]   # return a copy


# ─────────────────────────── crossover ────────────────────────────

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX1) — adapted for (box_index, rotation) tuples.

    How it works:
    1. Pick two random cut points to define a segment in parent1.
    2. Copy that segment directly into the child at the same positions.
    3. Fill the remaining positions with the genes from parent2,
       in the order they appear, skipping any box_index already in the child.
    4. Rotation indices travel with their box_index.

    Returns two children (symmetric: roles of parents swap for child2).
    """
    size = len(parent1)
    cx1, cx2 = sorted(random.sample(range(size), 2))

    def _build_child(p1, p2):
        child = [None] * size
        # Step 1 — copy the segment from p1
        child[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
        segment_indices = {gene[0] for gene in child[cx1:cx2 + 1]}

        # Step 2 — fill remaining slots from p2, preserving p2's order
        fill_values = [gene for gene in p2 if gene[0] not in segment_indices]
        fill_idx = 0
        for i in range(size):
            if child[i] is None:
                child[i] = fill_values[fill_idx]
                fill_idx += 1
        return child

    child1 = _build_child(parent1, parent2)
    child2 = _build_child(parent2, parent1)
    return child1, child2


# ─────────────────────────── mutation ─────────────────────────────

def swap_mutation(chromosome):
    """
    Swap Mutation: pick two random positions and swap their genes.
    Minimal disruption — changes only two elements.
    """
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome


def inversion_mutation(chromosome):
    """
    Inversion Mutation: pick a random sub-sequence and reverse it.
    More disruptive than a swap — helps escape local optima.
    """
    a, b = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[a:b + 1] = chromosome[a:b + 1][::-1]
    return chromosome


def rotation_mutation(chromosome):
    """
    Rotation Mutation: pick a random gene and change its rotation index.
    Explores different box orientations without changing the order.
    """
    idx = random.randint(0, len(chromosome) - 1)
    box_idx, _ = chromosome[idx]
    chromosome[idx] = (box_idx, random.randint(0, 5))
    return chromosome


def mutate(chromosome, mutation_swap_rate, mutation_inv_rate):
    """
    Apply mutation operators with independent (adaptive) probabilities.
    Each operator is applied independently, so a chromosome may receive
    no mutation, one, two, or all three in the same generation.
    """
    if random.random() < mutation_swap_rate:
        chromosome = swap_mutation(chromosome)
    if random.random() < mutation_inv_rate:
        chromosome = inversion_mutation(chromosome)
    if random.random() < MUTATION_ROT:
        chromosome = rotation_mutation(chromosome)
    return chromosome


# ─────────────────────────── main GA loop ─────────────────────────

def solve(boxes, container, verbose=True):
    """
    Run the full Genetic Algorithm and return the best packing found.

    Parameters
    ----------
    boxes : list[Box]
        All boxes to attempt to pack.
    container : Container
        The shipping container.
    verbose : bool
        If True, print progress every 10 generations.

    Returns
    -------
    dict with keys:
        "positions"   – list of placement dicts (the packing plan)
        "utilization" – best volume utilization in %
        "fitness"     – best fitness score
        "generation"  – generation in which the best was found
    """
    n = len(boxes)

    # ── 1. Initialize population ──────────────────────────────────
    population = []

    # Seed a few individuals with greedy order for a strong start
    num_greedy_seeds = min(3, POPULATION_SIZE)
    for i in range(num_greedy_seeds):
        ch = greedy_seeded_chromosome(n, boxes)
        # Add some random rotations to the seeded copies (except the first)
        if i > 0:
            for j in range(len(ch)):
                if random.random() < 0.3:
                    ch[j] = (ch[j][0], random.randint(0, 5))
        population.append(ch)

    # Fill the rest with random chromosomes
    for _ in range(POPULATION_SIZE - num_greedy_seeds):
        population.append(random_chromosome(n))

    # Evaluate the initial population
    eval_results = [fitness(ch, boxes, container) for ch in population]
    scores        = [r[0] for r in eval_results]
    all_positions = [r[1] for r in eval_results]
    all_utils     = [r[2] for r in eval_results]

    # Track the global best
    best_idx       = max(range(POPULATION_SIZE), key=lambda i: scores[i])
    best_score     = scores[best_idx]
    best_positions = all_positions[best_idx]
    best_util      = all_utils[best_idx]
    best_gen       = 0

    # Adaptive mutation state
    mutation_swap_rate = MUTATION_SWAP_BASE
    mutation_inv_rate  = MUTATION_INV_BASE
    gens_since_improvement = 0

    if verbose:
        print(f"[GA] gen   0 | best fitness {best_score:7.2f}% "
              f"| utilization {best_util:5.1f}% "
              f"| placed {len(best_positions)}/{n}")

    # ── 2. Evolve ─────────────────────────────────────────────────
    for gen in range(1, NUM_GENERATIONS + 1):

        # Sort current population by score (descending) for elitism
        ranked = sorted(range(POPULATION_SIZE),
                        key=lambda i: scores[i], reverse=True)

        new_population = []

        # 2a. Elitism — carry over the top ELITE_COUNT unchanged
        for i in range(ELITE_COUNT):
            new_population.append([gene for gene in population[ranked[i]]])

        # 2b. Fill the rest with crossover + mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)

            # Crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = [g for g in parent1], [g for g in parent2]

            # Mutation (with adaptive rates)
            child1 = mutate(child1, mutation_swap_rate, mutation_inv_rate)
            child2 = mutate(child2, mutation_swap_rate, mutation_inv_rate)

            new_population.append(child1)
            if len(new_population) < POPULATION_SIZE:
                new_population.append(child2)

        population = new_population

        # 2c. Evaluate the new generation
        eval_results  = [fitness(ch, boxes, container) for ch in population]
        scores        = [r[0] for r in eval_results]
        all_positions = [r[1] for r in eval_results]
        all_utils     = [r[2] for r in eval_results]

        # Update global best
        gen_best_idx = max(range(POPULATION_SIZE), key=lambda i: scores[i])
        if scores[gen_best_idx] > best_score:
            best_score     = scores[gen_best_idx]
            best_positions = all_positions[gen_best_idx]
            best_util      = all_utils[gen_best_idx]
            best_gen       = gen
            gens_since_improvement = 0
            # Reset mutation rates on improvement
            mutation_swap_rate = MUTATION_SWAP_BASE
            mutation_inv_rate  = MUTATION_INV_BASE
        else:
            gens_since_improvement += 1

        # 2d. Adaptive mutation: increase rates during stagnation
        if gens_since_improvement > 0 and gens_since_improvement % 10 == 0:
            mutation_swap_rate = min(mutation_swap_rate * 1.5, 0.50)
            mutation_inv_rate  = min(mutation_inv_rate * 1.5, 0.40)
            if verbose:
                print(f"[GA]   ^ mutation rates: swap={mutation_swap_rate:.2f}, "
                      f"inv={mutation_inv_rate:.2f}")

        # 2e. Stagnation immigration: replace bottom 20% with fresh individuals
        if gens_since_improvement == STAGNATION_LIMIT:
            num_immigrants = max(1, int(POPULATION_SIZE * IMMIGRATION_RATE))
            for i in range(num_immigrants):
                replace_idx = ranked[-(i + 1)]  # replace worst individuals
                population[replace_idx] = random_chromosome(n)
                # Re-evaluate the replaced individual
                s, p, u = fitness(population[replace_idx], boxes, container)
                scores[replace_idx] = s
                all_positions[replace_idx] = p
                all_utils[replace_idx] = u
            if verbose:
                print(f"[GA]   >> Immigration: replaced {num_immigrants} "
                      f"weakest individuals")

        # 2f. Early stopping
        if gens_since_improvement >= EARLY_STOP_GENS:
            if verbose:
                print(f"[GA]   >> Early stop at gen {gen} "
                      f"(no improvement for {EARLY_STOP_GENS} gens)")
            break

        if verbose and (gen % 10 == 0 or gen == NUM_GENERATIONS):
            print(f"[GA] gen {gen:3d} | best fitness {best_score:7.2f}% "
                  f"| utilization {best_util:5.1f}% "
                  f"| placed {len(best_positions)}/{n}")

    # ── 3. Return results ─────────────────────────────────────────
    if verbose:
        print(f"\n[GA] Done - best found at generation {best_gen}")
        print(f"[GA]   fitness    = {best_score:.2f}")
        print(f"[GA]   utilization = {best_util:.1f}%")
        print(f"[GA]   placed     = {len(best_positions)}/{n} boxes")

    return {
        "positions":   best_positions,
        "utilization": best_util,
        "fitness":     best_score,
        "generation":  best_gen,
    }


