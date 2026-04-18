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
 
 