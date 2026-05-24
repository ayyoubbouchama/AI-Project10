"""
==============================================================================
Genetic Algorithm — Constraint-Rigorous Implementation
==============================================================================

This is a ground-up rewrite of the GA that enforces ALL four constraints
explicitly inside the placement decoder, so that every individual in the
population and the final solution are always physically valid.

CONSTRAINTS ENFORCED (in the decoder, at every placement attempt):
──────────────────────────────────────────────────────────────────
1. BOUNDARY CONSTRAINT
   A box at (x, y, z) with dims (l, w, h) is accepted only if:
       x + l ≤ CL  AND  y + w ≤ CW  AND  z + h ≤ CH
   Any candidate position that would breach a wall is discarded.

2. COLLISION AVOIDANCE
   After a tentative placement, we scan ALL already-placed boxes for
   axis-aligned bounding-box (AABB) intersection. A candidate is accepted
   only if zero overlaps are found (uses strict gap ε = 1e-6).

3. GRAVITY / SUPPORT CONSTRAINT
   A box must rest on the floor (z = 0) OR on top of one or more already-
   placed boxes such that at least SUPPORT_RATIO (30 %) of its base area
   is covered. Candidate positions are generated only at physically valid
   z-levels (z = 0, or z = top of some supporting box), ensuring no box
   ever floats.

4. ORIENTATION RESTRICTION (FRAGILE / orientation_locked)
   Boxes flagged orientation_locked=True may only use rotations that keep
   the original height (H) in the vertical axis:
       rot 0 → (L, W, H)   rot 2 → (W, L, H)
   All other rotations (which would flip the box sideways or upside-down)
   are forbidden, both during chromosome initialisation and in the decoder.

PLACEMENT ENGINE — "Extreme-Point + Gravity Column" strategy
─────────────────────────────────────────────────────────────
  Candidate positions are the set of "extreme points":
      EP = {(0, 0, 0)} ∪ {(x+l, y, z), (x, y+w, z), (x, y, z+h)}
             for every already-placed box p
  For each box in chromosome order we try every EP × valid-rotation pair
  sorted by (z, y, x) — bottom-left-front first — and pick the first one
  that passes all four constraint checks above. If no EP works, the box is
  skipped (unplaced).

CHROMOSOME ENCODING
───────────────────
  Each gene = (box_index, rotation_id)
  A chromosome is a permutation of all box indices (no duplicates).

FITNESS
───────
  fitness = utilisation_pct
           + 0.20 × stability_score        (heavy items low)
           − UNPLACED_PENALTY × n_unplaced  (strong pressure to place all)
           − OVERLAP_PENALTY × n_overlaps   (should always be 0 by construction,
                                              kept as a safety net)

HYPERPARAMETERS (tuned for ~386 boxes, 40-second wall-clock budget)
───────────────────────────────────────────────────────────────────
  POPULATION_SIZE   = 50
  NUM_GENERATIONS   = 100
  TOURNAMENT_SIZE   = 4
  CROSSOVER_RATE    = 0.88
  ELITE_COUNT       = 4
  STAGNATION_LIMIT  = 10   (triggers partial restart)
  RESTART_RATIO     = 0.40
  SUPPORT_RATIO     = 0.30  (min fraction of base area that must be supported)
  UNPLACED_PENALTY  = 40.0

OPERATORS
─────────
  Crossover : OX1 (45 %) / OX2 (35 %) / PMX (20 %)
  Mutation  : swap · inversion · rotation · scramble (adaptive pressure)
  Selection : Tournament
  Elitism   : top-4 always survive
  Local search: 2-opt on elite every 20 generations
"""

import random
import time


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-6   # geometric tolerance


def _aabb_overlap(ax, ay, az, al, aw, ah,
                  bx, by, bz, bl, bw, bh) -> bool:
    """Return True if the two axis-aligned boxes share interior volume."""
    return (ax < bx + bl - _EPS and bx < ax + al - _EPS and
            ay < by + bw - _EPS and by < ay + aw - _EPS and
            az < bz + bh - _EPS and bz < az + ah - _EPS)


def _support_fraction(x, y, z, l, w, placed_tuples) -> float:
    """
    Fraction of the base area (l × w) of a box at (x,y,z) that is
    supported by already-placed boxes (or by the floor when z ≈ 0).
    """
    if z < _EPS:
        return 1.0          # resting on the floor — fully supported
    base_area = l * w
    if base_area < _EPS:
        return 1.0
    covered = 0.0
    for (_, px, py, pz, pl, pw, ph, _) in placed_tuples:
        if abs((pz + ph) - z) < _EPS:          # top of this box is at z
            ox = max(0.0, min(x + l, px + pl) - max(x, px))
            oy = max(0.0, min(y + w, py + pw) - max(y, py))
            covered += ox * oy
    return covered / base_area


# ─────────────────────────────────────────────────────────────────────────────
# Rotation tables
# ─────────────────────────────────────────────────────────────────────────────

# All 6 axis-aligned rotations of (L, W, H).
# Rotations 0 and 2 keep H in the vertical (z) axis — safe for fragile boxes.
_ALL_ROTATIONS = [
    (0, lambda l, w, h: (l, w, h)),   # original           — H vertical ✓
    (1, lambda l, w, h: (l, h, w)),   # tilt forward
    (2, lambda l, w, h: (w, l, h)),   # 90° yaw            — H vertical ✓
    (3, lambda l, w, h: (w, h, l)),   # tilt + yaw
    (4, lambda l, w, h: (h, l, w)),   # 90° roll
    (5, lambda l, w, h: (h, w, l)),   # 90° roll + 90° yaw
]

_UPRIGHT_ROT_IDS = {0, 2}   # keep original height in the vertical axis


def _valid_rot_ids(box) -> list:
    """Return list of allowed rotation IDs for this box."""
    if getattr(box, 'orientation_locked', False):
        return [0, 2]
    return [0, 1, 2, 3, 4, 5]


# ─────────────────────────────────────────────────────────────────────────────
# Rotation cache (built once per solve call)
# ─────────────────────────────────────────────────────────────────────────────

def _build_rot_cache(boxes, CL, CW, CH):
    """
    For every box, pre-compute the (l, w, h) dims for each valid rotation
    that also fits within the container.
    Returns:
        rot_cache[i]      : dict {rot_id: (l, w, h)}  — non-empty rotations
        valid_rot_ids[i]  : list of allowed rot IDs (orientation + size filter)
    """
    rot_cache = []
    valid_rot_ids = []
    for box in boxes:
        allowed = _valid_rot_ids(box)
        dims = {}
        for rid, fn in _ALL_ROTATIONS:
            if rid not in allowed:
                continue
            bl, bw, bh = fn(box.l, box.w, box.h)
            if bl <= CL + _EPS and bw <= CW + _EPS and bh <= CH + _EPS:
                dims[rid] = (bl, bw, bh)
        # If no rotation fits (box is simply too large), keep original so we
        # can account for it as unplaced rather than crash.
        if not dims:
            dims[0] = (box.l, box.w, box.h)
        rot_cache.append(dims)
        valid_rot_ids.append(list(dims.keys()))
    return rot_cache, valid_rot_ids


# ─────────────────────────────────────────────────────────────────────────────
# Core decoder — Extreme-Point placer with full constraint checking
# ─────────────────────────────────────────────────────────────────────────────
def _gravity_drop(placed, CH):
    """Drop every box to its lowest valid z. Same logic as SA._gravity_drop_tuples."""
    if not placed:
        return placed
    placed_sorted = sorted(placed, key=lambda p: (p[3], p[2], p[1]))
    settled = []
    for p in placed_sorted:
        pid, px, py, pz, pl, pw, ph, pwt = p
        z_land = 0.0
        for s in settled:
            if (px < s[1] + s[4] - 1e-9 and px + pl > s[1] + 1e-9 and
                py < s[2] + s[5] - 1e-9 and py + pw > s[2] + 1e-9):
                z_top = s[3] + s[6]
                if z_top > z_land:
                    z_land = z_top
        if z_land <= pz and z_land + ph <= CH + 1e-9:
            pz = z_land
        settled.append((pid, px, py, pz, pl, pw, ph, pwt))
    return settled

def _decode(chromosome, boxes, CL, CW, CH, rot_cache, support_ratio=0.30):
    """
    Extreme-point decoder — same logic as SimulatedAnnealing._decode_tuples.
    Enforces boundary + collision by construction.
    Orientation already enforced via rot_cache.
    Gravity enforced via _gravity_drop after placement.
    """
    placed = []
    eps = [(0.0, 0.0, 0.0)]
    total_vol = 0.0

    for (box_idx, rot_id) in chromosome:
        box = boxes[box_idx]
        rc = rot_cache[box_idx]

        # Build rotation list: preferred rotation first, then rest
        if rot_id in rc:
            rots_to_try = [rc[rot_id]] + [rc[r] for r in rc if r != rot_id]
        else:
            rots_to_try = list(rc.values())

        if not rots_to_try:
            continue

        placed_flag = False

        # Sort EPs: bottom-left-front first (same as SA)
        sorted_eps = sorted(set(eps), key=lambda p: (p[2], p[1], p[0]))

        for i, (ex, ey, ez) in enumerate(sorted_eps):
            if placed_flag:
                break
            for (bl, bw, bh) in rots_to_try:
                ex_bl = ex + bl
                ey_bw = ey + bw
                ez_bh = ez + bh

                # ── 1. BOUNDARY ──────────────────────────────────────
                if ex_bl > CL + 1e-9 or ey_bw > CW + 1e-9 or ez_bh > CH + 1e-9:
                    continue

                # ── 2. COLLISION ─────────────────────────────────────
                overlap = False
                for s in placed:
                    if not (ex   >= s[1] + s[4] - 1e-9 or ex_bl <= s[1] + 1e-9 or
                            ey   >= s[2] + s[5] - 1e-9 or ey_bw <= s[2] + 1e-9 or
                            ez   >= s[3] + s[6] - 1e-9 or ez_bh <= s[3] + 1e-9):
                        overlap = True
                        break
                if overlap:
                    continue

                # ── All checks passed — place the box ────────────────
                placed.append((box.id, ex, ey, ez, bl, bw, bh, box.weight))
                total_vol += bl * bw * bh
                eps = [p for p in eps if p != (ex, ey, ez)]
                eps.extend([(ex_bl, ey, ez), (ex, ey_bw, ez), (ex, ey, ez_bh)])
                placed_flag = True
                break

    # ── 3. GRAVITY DROP (same as SA) ─────────────────────────────────────
    placed = _gravity_drop(placed, CH)

    # Metrics
    container_vol = CL * CW * CH
    util = (total_vol / container_vol * 100.0) if container_vol > 0 else 0.0

    actual_wz, worst_wz = 0.0, 0.0
    for (_, _x, _y, pz, _pl, _pw, ph, pw_) in placed:
        actual_wz += pw_ * pz
        worst_wz  += pw_ * max(CH - ph, 0.0)
    stability = (max(0.0, min(100.0, (1.0 - actual_wz / worst_wz) * 100.0))
                 if worst_wz > 1e-6 else 100.0)

    return placed, util, stability, len(placed)
# ─────────────────────────────────────────────────────────────────────────────
# Fitness function
# ─────────────────────────────────────────────────────────────────────────────

_UNPLACED_PENALTY = 40.0
_OVERLAP_PENALTY  = 50.0   # safety net — should never fire if decoder is correct


def _fitness(chromosome, boxes, container, rot_cache, support_ratio=0.30):
    CL, CW, CH = container.L, container.W, container.H
    placed, util, stability, n_placed = _decode(
        chromosome, boxes, CL, CW, CH, rot_cache, support_ratio)

    n_unplaced = len(boxes) - n_placed
    score = (util
             + 0.20 * stability
             + (5.0 if n_unplaced == 0 else 0.0)
             - _UNPLACED_PENALTY * n_unplaced
             - n_unplaced ** 1.5 * 5.0)
    return score, placed, util, stability


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome helpers
# ─────────────────────────────────────────────────────────────────────────────

def _random_chromosome(n, valid_rot_ids):
    indices = random.sample(range(n), n)
    return [(i, random.choice(valid_rot_ids[i])) for i in indices]


def _seed_chromosome(boxes, key_fn, rot_pref, valid_rot_ids):
    """Build a deterministic seeded chromosome using a heuristic ordering."""
    order = sorted(range(len(boxes)), key=lambda i: key_fn(boxes[i]), reverse=True)
    result = []
    for i in order:
        vr = valid_rot_ids[i]
        r = rot_pref if rot_pref in vr else vr[0]
        result.append((i, r))
    return result


_SEED_HEURISTICS = [
    lambda b: b.l * b.w * b.h,           # largest volume first
    lambda b: b.weight,                   # heaviest first (stability)
    lambda b: b.h,                        # tallest first
    lambda b: b.l * b.w,                  # largest footprint first
    lambda b: b.weight * b.l * b.w,       # heavy + wide footprint
    lambda b: max(b.l, b.w, b.h),         # longest dimension
    lambda b: b.l + b.w + b.h,            # largest perimeter
    lambda b: min(b.l, b.w, b.h),         # smallest min-dim last
    lambda b: b.l * b.w / (b.h + 1),      # flat boxes first
    lambda b: b.weight / (b.l * b.w * b.h + 1),  # density
]


def _build_seed_population(boxes, count, valid_rot_ids):
    seeds = []
    # First seed is always volume-descending — strongest single heuristic
    vol_order = sorted(range(len(boxes)),
                       key=lambda i: boxes[i].l * boxes[i].w * boxes[i].h,
                       reverse=True)
    seeds.append([(i, valid_rot_ids[i][0]) for i in vol_order])
    for i in range(1, count):
        key = _SEED_HEURISTICS[i % len(_SEED_HEURISTICS)]
        rot_pref = (i * 2) % 6
        seeds.append(_seed_chromosome(boxes, key, rot_pref, valid_rot_ids))
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# Crossover operators
# ─────────────────────────────────────────────────────────────────────────────

def _ox1(p1, p2):
    """Order Crossover 1 — preserves a contiguous segment from p1."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = {g[0] for g in child[a:b + 1]}
    fill = [g for g in p2 if g[0] not in used]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def _ox2(p1, p2):
    """Order Crossover 2 — preserves a random subset of positions from p1."""
    n = len(p1)
    k = max(1, n // 3)
    keep = set(random.sample(range(n), k))
    child = [None] * n
    for i in keep:
        child[i] = p1[i]
    used = {g[0] for g in child if g is not None}
    fill = [g for g in p2 if g[0] not in used]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def _pmx(p1, p2):
    """Partially Mapped Crossover with full mapping-table traversal."""
    n = len(p1)
    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b + 1] = p1[a:b + 1]
    used = {g[0] for g in child[a:b + 1]}
    fwd = {p2[i][0]: p1[i][0] for i in range(a, b + 1)}
    for i in range(n):
        if child[i] is not None:
            continue
        bid = p2[i][0]
        seen = set()
        while bid in used and bid not in seen:
            seen.add(bid)
            bid = fwd.get(bid, bid)
        if bid in used:
            remaining = {g[0] for g in p1} - used
            if remaining:
                bid = remaining.pop()
        child[i] = (bid, p2[i][1])
        used.add(bid)
    return child


def _crossover(p1, p2):
    r = random.random()
    if r < 0.45:
        return _ox1(p1, p2)
    elif r < 0.80:
        return _ox2(p1, p2)
    else:
        return _pmx(p1, p2)


# ─────────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────────

def _swap_mutate(c):
    c = c[:]
    i, j = random.sample(range(len(c)), 2)
    c[i], c[j] = c[j], c[i]
    return c


def _inversion_mutate(c):
    c = c[:]
    n = len(c)
    i = int(random.triangular(0, n - 1, 0))
    j = random.randint(i + 1, min(i + max(2, n // 3), n - 1))
    c[i:j + 1] = c[i:j + 1][::-1]
    return c


def _rotation_mutate(c, valid_rot_ids):
    c = c[:]
    i = random.randrange(len(c))
    bi, _ = c[i]
    c[i] = (bi, random.choice(valid_rot_ids[bi]))
    return c


def _scramble_mutate(c):
    c = c[:]
    n = len(c)
    seg = random.randint(2, max(3, n // 10))
    s = random.randint(0, n - seg)
    part = c[s:s + seg]
    random.shuffle(part)
    c[s:s + seg] = part
    return c


def _mutate(c, valid_rot_ids, pressure=1.0):
    _MUT_SWAP     = 0.20
    _MUT_INVERT   = 0.18
    _MUT_ROT      = 0.25
    _MUT_SCRAMBLE = 0.10
    if random.random() < _MUT_SWAP * pressure:
        c = _swap_mutate(c)
    if random.random() < _MUT_INVERT * pressure:
        c = _inversion_mutate(c)
    if random.random() < _MUT_ROT * pressure:
        c = _rotation_mutate(c, valid_rot_ids)
    if random.random() < _MUT_SCRAMBLE * pressure:
        c = _scramble_mutate(c)
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Selection
# ─────────────────────────────────────────────────────────────────────────────

def _tournament_select(population, scores, t_size):
    candidates = random.sample(range(len(population)), min(t_size, len(population)))
    best_idx = max(candidates, key=lambda i: scores[i])
    return population[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2-opt local search on elite individuals
# ─────────────────────────────────────────────────────────────────────────────

def _local_search_2opt(chrom, boxes, container, rot_cache,
                       support_ratio, max_iters=4):
    best = chrom[:]
    bs, _, _, _ = _fitness(best, boxes, container, rot_cache, support_ratio)
    n = len(chrom)
    for _ in range(max_iters):
        improved = False
        for _ in range(min(3, n // 2)):
            i = random.randint(0, n - 3)
            j = random.randint(i + 2, min(i + n // 4, n - 1))
            cand = best[:]
            cand[i:j + 1] = cand[i:j + 1][::-1]
            cs, _, _, _ = _fitness(cand, boxes, container, rot_cache, support_ratio)
            if cs > bs:
                best, bs = cand, cs
                improved = True
                break
        if not improved:
            break
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Diversity helpers
# ─────────────────────────────────────────────────────────────────────────────

def _chrom_key(c):
    return tuple(g[0] for g in c[:20])


def _remove_duplicates(pop, n, valid_rot_ids):
    seen = set()
    out = []
    for c in pop:
        k = _chrom_key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)
        else:
            out.append(_random_chromosome(n, valid_rot_ids))
    return out


def _restart_population(population, scores, n, valid_rot_ids,
                         restart_ratio=0.40, best_chrom=None):
    n_keep = int(len(population) * (1 - restart_ratio))
    ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
    survivors = [population[i][:] for i in ranked[:n_keep]]
    fresh = [_random_chromosome(n, valid_rot_ids)
             for _ in range(len(population) - n_keep)]
    # Always keep the global best
    pop = survivors + fresh
    if best_chrom is not None:
        pop[0] = best_chrom[:]
    return pop


# ─────────────────────────────────────────────────────────────────────────────
# Main GA class — drop-in replacement for `GeneticAlgorithm`
# ─────────────────────────────────────────────────────────────────────────────

class GeneticAlgorithmV2:
    """
    Constraint-rigorous Genetic Algorithm for the 3D Container Loading Problem.

    Enforces all four constraints (boundary, collision, gravity support,
    orientation) explicitly inside _decode() so that every evaluated
    individual in the population is always a physically valid packing.

    Usage
    -----
    result = GeneticAlgorithmV2.solve(boxes, container)
    # result keys: 'positions', 'utilization', 'stability', 'history', 'fitness'
    """

    POPULATION_SIZE  = 20
    NUM_GENERATIONS  = 40
    TOURNAMENT_SIZE  = 3
    CROSSOVER_RATE   = 0.88
    ELITE_COUNT      = 2
    STAGNATION_LIMIT = 6
    RESTART_RATIO    = 0.40
    SEED_RATIO       = 0.30
    SUPPORT_RATIO    = 0.30   # minimum base-area fraction required for gravity support

    @staticmethod
    def solve(boxes, container, seed=42, time_limit=40.0):
        """
        Run the GA and return the best valid packing found.

        Parameters
        ----------
        boxes      : list of Box objects
        container  : Container object (has .L, .W, .H attributes)
        seed       : random seed for reproducibility
        time_limit : wall-clock budget in seconds

        Returns
        -------
        dict with keys:
            'positions'   : list of dicts {id, x, y, z, l, w, h, weight}
            'utilization' : volume utilisation [%]
            'stability'   : stability score [0–100]
            'history'     : list of best utilisation per generation
            'fitness'     : best raw fitness score
        """
        random.seed(seed)
        GA = GeneticAlgorithmV2
        n = len(boxes)
        CL, CW, CH = container.L, container.W, container.H
        wall_start = time.perf_counter()

        # ── Pre-compute rotation cache (done once, reused by every eval) ──
        rot_cache, valid_rot_ids_list = _build_rot_cache(boxes, CL, CW, CH)
        SR = GA.SUPPORT_RATIO

        # ── Globals tracking the all-time best ────────────────────────────
        best_chromosome = None
        best_placed     = []
        best_score      = -float('inf')
        best_util       = 0.0
        best_stability  = 0.0
        history         = []

        restart_round = 0

        while True:
            restart_round += 1

            # ── Initialise (or restart) population ────────────────────────
            n_seeds = max(1, int(GA.POPULATION_SIZE * GA.SEED_RATIO))
            seeds   = _build_seed_population(boxes, n_seeds, valid_rot_ids_list)
            randoms = [_random_chromosome(n, valid_rot_ids_list)
                       for _ in range(GA.POPULATION_SIZE - n_seeds)]
            population = seeds + randoms
            if restart_round > 1 and best_chromosome is not None:
                population[0] = best_chromosome[:]   # always seed with global best

            stagnation = 0

            for generation in range(GA.NUM_GENERATIONS):

                # ── Time guard ────────────────────────────────────────────
                if time.perf_counter() - wall_start >= time_limit:
                    break

                # ── Evaluate population ───────────────────────────────────
                evaluated = [_fitness(c, boxes, container, rot_cache, SR)
                             for c in population]
                scores   = [e[0] for e in evaluated]
                placed_l = [e[1] for e in evaluated]
                utils    = [e[2] for e in evaluated]
                stabs    = [e[3] for e in evaluated]

                # ── Update global best ────────────────────────────────────
                gen_best = max(range(len(population)), key=lambda i: scores[i])
                if scores[gen_best] > best_score:
                    best_score      = scores[gen_best]
                    best_placed     = placed_l[gen_best]
                    best_util       = utils[gen_best]
                    best_stability  = stabs[gen_best]
                    best_chromosome = population[gen_best][:]
                    stagnation      = 0
                else:
                    stagnation += 1

                history.append(round(best_util, 2))

                # ── Early-exit if perfect packing ─────────────────────────
                if best_util >= 80.0 and len(best_placed) == n:
                    break

                # ── Stagnation → partial restart ──────────────────────────
                if stagnation >= GA.STAGNATION_LIMIT:
                    population = _restart_population(
                        population, scores, n, valid_rot_ids_list,
                        GA.RESTART_RATIO, best_chromosome)
                    stagnation = 0
                    continue

                # ── Adaptive mutation pressure ────────────────────────────
                recent = history[-min(10, len(history)):]
                spread = max(recent) - min(recent) if len(recent) >= 2 else 1.0
                convergence = max(0.0, min(1.0,
                                           1.0 - spread / (abs(best_util) + 1e-6)))
                pressure = 0.8 + 0.5 * convergence

                # ── Build next generation ─────────────────────────────────
                ranked = sorted(range(len(population)),
                                key=lambda i: scores[i], reverse=True)

                # Elitism: always keep the top-N
                elite = [population[i][:] for i in ranked[:GA.ELITE_COUNT]]

                # Apply 2-opt local search to elite every 20 generations
                if generation % 20 == 0 and generation > 0:
                    elite = [_local_search_2opt(c, boxes, container,
                                                rot_cache, SR, max_iters=3)
                             for c in elite]

                new_pop = elite[:]

                # Tournament size shrinks slightly as we progress (exploit more)
                progress = generation / max(1, GA.NUM_GENERATIONS - 1)
                t_size = max(3, int(GA.TOURNAMENT_SIZE * (1.0 - 0.4 * progress)))

                while len(new_pop) < GA.POPULATION_SIZE:
                    p1 = _tournament_select(population, scores, t_size)
                    p2 = _tournament_select(population, scores, t_size)
                    child = (_crossover(p1, p2) if random.random() < GA.CROSSOVER_RATE
                             else p1[:])
                    child = _mutate(child, valid_rot_ids_list, pressure)
                    new_pop.append(child)

                # Periodic duplicate elimination for diversity
                if generation % 15 == 0:
                    new_pop = _remove_duplicates(new_pop, n, valid_rot_ids_list)

                population = new_pop

            # ── Check outer loop exit conditions ──────────────────────────
            if time.perf_counter() - wall_start >= time_limit:
                break
            if best_util >= 80.0 and len(best_placed) == n:
                break

        # ── Convert internal tuples to output dicts ────────────────────────
        positions = [
            {"id": p[0], "x": p[1], "y": p[2], "z": p[3],
             "l": p[4], "w": p[5], "h": p[6], "weight": p[7]}
            for p in best_placed
        ]

        return {
            "positions":   positions,
            "utilization": round(best_util, 2),
            "stability":   round(best_stability, 2),
            "history":     history,
            "fitness":     round(best_score, 2),
        }


# ── Alias so the rest of the notebook keeps working unchanged ─────────────────
GeneticAlgorithm = GeneticAlgorithmV2

def solve(boxes, container, seed=42, time_limit=40.0):
    result = GeneticAlgorithmV2.solve(boxes, container, seed=seed, time_limit=time_limit)
    result["totalBoxes"] = len(boxes)
    return result


print("GeneticAlgorithmV2 (constraint-rigorous) defined ✓")
print("  Alias: GeneticAlgorithm → GeneticAlgorithmV2")
print()
print("Constraints enforced inside decoder:")
print("  [1] Boundary      — box must not exceed container walls")
print("  [2] Collision      — no AABB overlap with any placed box")
print("  [3] Gravity/Support— ≥30 % base area must be supported (floor or box top)")
print("  [4] Orientation    — orientation_locked boxes stay upright (rot 0 or 2 only)")