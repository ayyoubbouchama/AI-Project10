"""
Genetic Algorithm for 3D Container Loading (Bin Packing)
=========================================================
 
Key design decisions (explained for the report):
-------------------------------------------------
CHROMOSOME:
    A permutation of box indices, e.g. [2, 0, 4, 1, 3].
    The ORDER of boxes determines how the placer fills the container.
    The GA evolves better orderings, not positions directly.
 
FITNESS:
    fitness = utilization (%) - heavy_penalty * unplaced_count
    We reward high utilization and penalize any box that didn't fit.
    This keeps hard constraint violations (unplaced boxes) costly.
 
SELECTION:
    Tournament selection: pick k random individuals, keep the best.
    More robust than roulette wheel for noisy fitness landscapes.
 
CROSSOVER:
    Order Crossover (OX1): preserves relative ordering of elements.
    Required for permutation chromosomes (classic TSP/bin packing choice).
 
MUTATION:
    Two operators applied with independent probabilities:
    - Swap mutation:    swap two random positions.
    - Inversion mutation: reverse a random sub-sequence.
    Using both gives more search diversity.
 
ELITISM:
    The best individual always survives to the next generation.
    Prevents accidental loss of the best solution found.
"""
 
import random
from core.space_manager import place_boxes
from core.utils import compute_utilization
 
 
# ─────────────────────────── constants ────────────────────────────
 
POPULATION_SIZE  = 60       # number of individuals per generation
NUM_GENERATIONS  = 200      # how many generations to run
TOURNAMENT_SIZE  = 5        # candidates per tournament
CROSSOVER_RATE   = 0.85     # probability of crossover vs. direct copy
MUTATION_SWAP    = 0.15     # probability of swap mutation per individual
MUTATION_INVERT  = 0.10     # probability of inversion mutation per individual
UNPLACED_PENALTY = 5.0      # % penalty per unplaced box
ELITE_COUNT      = 2        # top N individuals copied unchanged each gen
 
 
# ─────────────────────────── chromosome ───────────────────────────
 
def random_chromosome(n):
    """A chromosome is a random permutation of box indices [0..n-1]."""
    c = list(range(n))
    random.shuffle(c)
    return c
 
 
# ─────────────────────────── fitness ──────────────────────────────
 
def fitness(chromosome, boxes, container):
    """
    Decode the chromosome into a packing, then score it.
 
    Score = utilization(%) - UNPLACED_PENALTY * #unplaced_boxes
 
    Higher is better. Maximum possible ≈ 100.0 (perfect fit, all placed).
    """
    ordered_boxes = [boxes[i] for i in chromosome]
    positions = place_boxes(ordered_boxes, container)
 
    utilization = compute_utilization(positions, container)
    unplaced = len(boxes) - len(positions)
    score = utilization - UNPLACED_PENALTY * unplaced
 
    return score, positions, utilization
 
 
# ─────────────────────────── selection ────────────────────────────
 
def tournament_select(population, scores, k=TOURNAMENT_SIZE):
    """
    Pick k individuals at random, return the one with the highest score.
    """
    candidates = random.sample(range(len(population)), k)
    best = max(candidates, key=lambda i: scores[i])
    return population[best][:]   # return a copy


# ─────────────────────────── crossover ────────────────────────────

def order_crossover(parent1, parent2):
    """
    Order Crossover (OX1) — the standard crossover for permutation
    representations (used in TSP, bin packing, scheduling, etc.).

    How it works:
    1. Pick two random cut points to define a segment in parent1.
    2. Copy that segment directly into the child at the same positions.
    3. Fill the remaining positions with the genes from parent2,
       in the order they appear, skipping any gene already in the child.

    This preserves the *relative order* of genes from parent2 while
    keeping an *absolute sub-sequence* from parent1.

    Returns two children (symmetric: roles of parents swap for child2).
    """
    size = len(parent1)
    # Choose two distinct cut points and sort them
    cx1, cx2 = sorted(random.sample(range(size), 2))

    def _build_child(p1, p2):
        child = [None] * size
        # Step 1 — copy the segment from p1
        child[cx1:cx2 + 1] = p1[cx1:cx2 + 1]
        segment_set = set(child[cx1:cx2 + 1])

        # Step 2 — fill remaining slots from p2, preserving p2's order
        fill_values = [g for g in p2 if g not in segment_set]
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

    Minimal disruption — changes only two elements. Good for fine-tuning
    an already-decent solution.
    """
    a, b = random.sample(range(len(chromosome)), 2)
    chromosome[a], chromosome[b] = chromosome[b], chromosome[a]
    return chromosome


def inversion_mutation(chromosome):
    """
    Inversion Mutation: pick a random sub-sequence and reverse it.

    More disruptive than a swap — can rearrange several elements at
    once, which helps escape local optima. The combination of swap +
    inversion gives the GA a balance of exploitation vs. exploration.
    """
    a, b = sorted(random.sample(range(len(chromosome)), 2))
    chromosome[a:b + 1] = chromosome[a:b + 1][::-1]
    return chromosome


def mutate(chromosome):
    """
    Apply mutation operators with independent probabilities.

    Each operator is applied independently, so a chromosome may receive
    no mutation, one mutation, or both in the same generation.
    """
    if random.random() < MUTATION_SWAP:
        chromosome = swap_mutation(chromosome)
    if random.random() < MUTATION_INVERT:
        chromosome = inversion_mutation(chromosome)
    return chromosome


# ─────────────────────────── main GA loop ─────────────────────────

def solve(boxes, container):
    """
    Run the full Genetic Algorithm and return the best packing found.

    Parameters
    ----------
    boxes : list[Box]
        All boxes to attempt to pack.
    container : Container
        The shipping container.

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
    population = [random_chromosome(n) for _ in range(POPULATION_SIZE)]

    # Evaluate the initial population
    eval_results = [fitness(ch, boxes, container) for ch in population]
    scores       = [r[0] for r in eval_results]
    all_positions = [r[1] for r in eval_results]
    all_utils     = [r[2] for r in eval_results]

    # Track the global best
    best_idx   = max(range(POPULATION_SIZE), key=lambda i: scores[i])
    best_score = scores[best_idx]
    best_positions = all_positions[best_idx]
    best_util  = all_utils[best_idx]
    best_gen   = 0



    # ── 2. Evolve ─────────────────────────────────────────────────
    for gen in range(1, NUM_GENERATIONS + 1):

        # Sort current population by score (descending) for elitism
        ranked = sorted(range(POPULATION_SIZE),
                        key=lambda i: scores[i], reverse=True)

        new_population = []

        # 2a. Elitism — carry over the top ELITE_COUNT unchanged
        for i in range(ELITE_COUNT):
            new_population.append(population[ranked[i]][:])

        # 2b. Fill the rest with crossover + mutation
        while len(new_population) < POPULATION_SIZE:
            parent1 = tournament_select(population, scores)
            parent2 = tournament_select(population, scores)

            # Crossover
            if random.random() < CROSSOVER_RATE:
                child1, child2 = order_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # Mutation
            child1 = mutate(child1)
            child2 = mutate(child2)

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



    # ── 3. Return results ─────────────────────────────────────────


    return {
        "positions":   best_positions,
        "utilization": best_util,
        "fitness":     best_score,
        "generation":  best_gen,
    }
 
 