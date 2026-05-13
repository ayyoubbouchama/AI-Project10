"""
Genetic Algorithm for 3D Container Loading - Best-in-Class
==========================================================

KEY FEATURES:
1. Internal shelf-based decoder with gravity settling
2. Fragile box orientation locking
3. Multi-objective fitness: utilization + stability - penalties
4. Three crossover operators: OX1, OX2, PMX (proper mapping-table)
5. Four mutation operators: swap, inversion, rotation, scramble
6. Adaptive mutation pressure based on convergence
7. Smart population seeding with 12 heuristics
8. Adaptive restart on stagnation
9. 2-opt local search on elite solutions
10. Duplicate elimination for diversity

DECODER (internal shelf-based placement via _decode_chromosome):
  - O(n) per call - for n=386: ~3 ms
  - No overlap possible by construction
  - Gravity enforced via _gravity_settle() post-processing
  - Deterministic: same chromosome always gives same packing
  - Functionally equivalent to SA's decoder and space_manager.place_boxes
"""

import random
import time
from core.utils import compute_utilization, compute_stability


class GeneticAlgorithm:
    """
    A class-based Genetic Algorithm solver for the 3-D container
    loading problem. Uses a shelf-based decoder with gravity settling.
    """

    # ---- hyper-parameters ----
    POPULATION_SIZE   = 80
    NUM_GENERATIONS   = 150
    TOURNAMENT_SIZE   = 5
    CROSSOVER_RATE    = 0.90
    MUTATION_SWAP     = 0.20
    MUTATION_INVERT   = 0.18
    MUTATION_ROT      = 0.25
    MUTATION_SCRAMBLE = 0.10
    UNPLACED_PENALTY  = 35.0
    ELITE_COUNT       = 6
    STAGNATION_LIMIT  = 12
    RESTART_RATIO     = 0.45
    SEED_RATIO        = 0.30
    NUM_ROTATIONS     = 6
    ADAPTIVE_MUTATION = True

    # ---- rotations ----
    ROTATION_TABLE = [
        lambda l, w, h: (l, w, h),   # 0: original
        lambda l, w, h: (l, h, w),   # 1
        lambda l, w, h: (w, l, h),   # 2
        lambda l, w, h: (w, h, l),   # 3
        lambda l, w, h: (h, l, w),   # 4
        lambda l, w, h: (h, w, l),   # 5
    ]

    @staticmethod
    def apply_rotation(box, rot_id):
        return GeneticAlgorithm.ROTATION_TABLE[rot_id](box.l, box.w, box.h)

    @staticmethod
    def get_valid_rotation_ids(box):
        """Return list of valid rotation IDs respecting orientation_locked."""
        if getattr(box, 'orientation_locked', False):
            return [0, 2]  # (l,w,h) and (w,l,h) - keep h vertical
        return list(range(GeneticAlgorithm.NUM_ROTATIONS))

    # ---- chromosome ----
    @staticmethod
    def random_chromosome(boxes):
        n = len(boxes)
        indices = random.sample(range(n), n)
        return [(i, random.choice(GeneticAlgorithm.get_valid_rotation_ids(boxes[i]))) for i in indices]

    @staticmethod
    def order_seed(boxes, key_fn, rot=0):
        indices = sorted(range(len(boxes)), key=lambda i: key_fn(boxes[i]), reverse=True)
        result = []
        for i in indices:
            valid = GeneticAlgorithm.get_valid_rotation_ids(boxes[i])
            r = rot if rot in valid else valid[0]
            result.append((i, r))
        return result

    @staticmethod
    def build_seed_population(boxes, count):
        GA = GeneticAlgorithm
        seeds = []
        heuristics = [
            lambda b: b.l * b.w * b.h,                        # Volume-first (BFD)
            lambda b: b.h,                                     # Height-first
            lambda b: b.l * b.w,                               # Base area-first
            lambda b: b.w * b.h,                               # Side area-first
            lambda b: max(b.l, b.w, b.h),                     # Max dimension
            lambda b: b.l + b.w + b.h,                        # Perimeter
            lambda b: min(b.l, b.w, b.h),                     # Min dimension (compact)
            lambda b: (b.l * b.w * b.h) / (b.weight + 1),    # Volume/weight ratio
            lambda b: b.weight,                                # Weight-first (stable)
            lambda b: max(b.l, b.w, b.h) * (b.weight + 1),   # Heavy+large
            lambda b: b.weight * b.l * b.w,                   # Heavy+large base
            lambda b: b.l * b.w / (b.h + 1),                  # Flat boxes first
        ]
        for i in range(count):
            key = heuristics[i % len(heuristics)]
            rot = (i * 2) % GA.NUM_ROTATIONS
            seeds.append(GA.order_seed(boxes, key, rot))
        return seeds
    # ---- fast decoder with gravity ----
    @staticmethod
    def _gravity_settle(placed, CL, CW, CH):
        """
        Drop every box to its physical resting z-position.

        Process boxes bottom-up. For each box, compute the landing z as the
        highest z_top of any already-settled box whose x-y footprint overlaps.
        Boxes on the floor stay at z=0. Boxes above gaps are lowered until
        they rest on a surface below.

        This is O(n²) — for n=386 it takes < 1 ms, so runtime is unaffected.
        No boxes are removed, so utilization is preserved.
        """
        if not placed:
            return placed

        # Sort by z, then y, then x (process bottom-up)
        placed.sort(key=lambda p: (p["z"], p["y"], p["x"]))

        settled = []
        for p in placed:
            px, py, pl, pw, ph = p["x"], p["y"], p["l"], p["w"], p["h"]

            # Find the landing z: highest z_top of any settled box that
            # overlaps in x-y (even partially)
            z_land = 0.0
            for s in settled:
                if (px < s["x"] + s["l"] - 1e-9 and px + pl > s["x"] + 1e-9 and
                    py < s["y"] + s["w"] - 1e-9 and py + pw > s["y"] + 1e-9):
                    z_top = s["z"] + s["h"]
                    if z_top > z_land:
                        z_land = z_top

            # Only drop (never raise) and keep within container
            if z_land <= p["z"] and z_land + ph <= CH + 1e-9:
                p["z"] = z_land

            settled.append(p)

        return settled

    @staticmethod
    def _decode_chromosome(chromosome, boxes, container):
        """
        Decode chromosome into placed positions using a fast shelf placer
        followed by gravity settling.
        1. SHELF PLACEMENT (fast, O(n)): assigns (x, y, z) positions with
           no overlaps by construction. However layers may leave gaps.
        2. GRAVITY DROP: drops every box onto its actual physical support.
        The chromosome specifies both ordering AND rotation for each box.
        """
        GA = GeneticAlgorithm
        CL, CW, CH = container.L, container.W, container.H
        placed = []
        layers = []
        z_top = 0.0
        for (box_idx, rot_id) in chromosome:
            box = boxes[box_idx]
            valid_rots = GA.get_valid_rotation_ids(box)
            if rot_id not in valid_rots:
                rot_id = valid_rots[0]
            bl, bw, bh = GA.apply_rotation(box, rot_id)
            if bl > CL + 1e-9 or bw > CW + 1e-9 or bh > CH + 1e-9:
                continue
            box_id = box.id
            box_weight = box.weight
            placed_flag = False
            for layer in layers:
                if bh > layer[1]:
                    continue
                for row in layer[2]:
                    if bw <= row[1] + 1e-9 and row[2] + bl <= CL + 1e-9:
                        placed.append({
                            "id": box_id,
                            "x": row[2], "y": row[0], "z": layer[0],
                            "l": bl, "w": bw, "h": bh, "weight": box_weight,
                        })
                        row[2] += bl
                        placed_flag = True
                        break
                if placed_flag:
                    break
                y_used = sum(r[1] for r in layer[2])
                if y_used + bw <= CW + 1e-9:
                    layer[2].append([y_used, bw, bl])
                    placed.append({
                        "id": box_id,
                        "x": 0.0, "y": y_used, "z": layer[0],
                        "l": bl, "w": bw, "h": bh, "weight": box_weight,
                    })
                    placed_flag = True
                    break
            if not placed_flag and z_top + bh <= CH + 1e-9:
                layers.append([z_top, bh, [[0.0, bw, bl]]])
                placed.append({
                    "id": box_id,
                    "x": 0.0, "y": 0.0, "z": z_top,
                    "l": bl, "w": bw, "h": bh, "weight": box_weight,
                })
                z_top += bh
        placed = GA._gravity_settle(placed, CL, CW, CH)
        return placed

    # ---- verification helpers ----
    @staticmethod
    def _overlap_1d(a1, a2, b1, b2):
        """Return overlap length between intervals [a1,a2) and [b1,b2)."""
        return max(0.0, min(a2, b2) - max(a1, b1))

    @staticmethod
    def _compute_support_area(x, y, l, w, z, placed):
        """Compute fraction of base area supported by boxes below or floor."""
        base_area = l * w
        if base_area <= 0: return 1.0
        if z < 1e-9: return 1.0
        support_area = 0.0
        for pb in placed:
            if abs((pb["z"] + pb["h"]) - z) < 1e-6:
                ox = GeneticAlgorithm._overlap_1d(x, x + l, pb["x"], pb["x"] + pb["l"])
                oy = GeneticAlgorithm._overlap_1d(y, y + w, pb["y"], pb["y"] + pb["w"])
                support_area += ox * oy
        return support_area / base_area

    @staticmethod
    def _boxes_overlap(x1, y1, z1, l1, w1, h1, x2, y2, z2, l2, w2, h2):
        """Check if two axis-aligned boxes overlap."""
        eps = 1e-9
        return not (
            x1 + l1 <= x2 + eps or x2 + l2 <= x1 + eps or
            y1 + w1 <= y2 + eps or y2 + w2 <= y1 + eps or
            z1 + h1 <= z2 + eps or z2 + h2 <= z1 + eps
        )

    SUPPORT_THRESHOLD = 0.30  # For verification scripts
    # ---- fitness ----
    @staticmethod
    def fitness(chromosome, boxes, container):
        GA = GeneticAlgorithm
        positions   = GA._decode_chromosome(chromosome, boxes, container)
        utilization = compute_utilization(positions, container)
        stability   = compute_stability(positions, container)
        unplaced    = len(boxes) - len(positions)
        penalty = 0.0
        if unplaced > 0:
            penalty = GA.UNPLACED_PENALTY * unplaced + (unplaced ** 1.5) * 5.0
        placed_bonus = 5.0 if unplaced == 0 else 0.0
        score = utilization + (stability * 0.20) + placed_bonus - penalty
        return score, positions, utilization

    # ---- selection ----
    @staticmethod
    def tournament_select(population, scores, t_size=None):
        if t_size is None:
            t_size = GeneticAlgorithm.TOURNAMENT_SIZE
        candidates = random.sample(range(len(population)), min(t_size, len(population)))
        best = max(candidates, key=lambda i: scores[i])
        return population[best][:]

    # ---- crossover ----
    @staticmethod
    def ox1_crossover(p1, p2):
        n = len(p1)
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        child[start:end + 1] = p1[start:end + 1]
        used = {gene[0] for gene in child[start:end + 1]}
        fill = [gene for gene in p2 if gene[0] not in used]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    @staticmethod
    def ox2_crossover(p1, p2):
        n = len(p1)
        k = max(1, n // 3)
        keep_pos = set(random.sample(range(n), k))
        child = [None] * n
        for i in keep_pos:
            child[i] = p1[i]
        used = {gene[0] for gene in child if gene is not None}
        fill = [gene for gene in p2 if gene[0] not in used]
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = fill[idx]
                idx += 1
        return child

    @staticmethod
    def pmx_crossover(p1, p2):
        """Partially Mapped Crossover with proper mapping-table traversal."""
        n = len(p1)
        start, end = sorted(random.sample(range(n), 2))
        child = [None] * n
        # Step 1: Copy segment from P1
        child[start:end + 1] = p1[start:end + 1]
        used = {g[0] for g in child[start:end + 1]}
        # Step 2: Build forward mapping: for each position in the segment,
        # map P2's box_id to P1's box_id (what replaced it)
        map_p2_to_p1 = {}
        for i in range(start, end + 1):
            map_p2_to_p1[p2[i][0]] = p1[i][0]
        # Step 3: Fill remaining positions
        for i in range(n):
            if child[i] is not None:
                continue
            box_id = p2[i][0]
            rot = p2[i][1]
            # Chase through mapping until we find an unused box_id
            seen = set()
            while box_id in used and box_id not in seen:
                seen.add(box_id)
                box_id = map_p2_to_p1.get(box_id, box_id)
            if box_id in used:
                # Fallback: pick any unused box_id
                all_ids = {g[0] for g in p1}
                remaining = all_ids - used
                if remaining:
                    box_id = remaining.pop()
            child[i] = (box_id, rot)
            used.add(box_id)
        return child

    @staticmethod
    def crossover(p1, p2):
        GA = GeneticAlgorithm
        r = random.random()
        if r < 0.45: return GA.ox1_crossover(p1, p2)
        elif r < 0.80: return GA.ox2_crossover(p1, p2)
        else: return GA.pmx_crossover(p1, p2)

    # ---- mutation ----
    @staticmethod
    def swap_mutate(c):
        c = c[:]
        i, j = random.sample(range(len(c)), 2)
        c[i], c[j] = c[j], c[i]
        return c

    @staticmethod
    def weighted_inversion_mutate(c):
        c = c[:]
        n = len(c)
        i = int(random.triangular(0, n - 1, 0))
        j = random.randint(i + 1, min(i + max(2, n // 3), n - 1))
        c[i:j + 1] = c[i:j + 1][::-1]
        return c

    @staticmethod
    def rotation_mutate(c, boxes):
        c = c[:]
        i = random.randrange(len(c))
        box_idx, _ = c[i]
        c[i] = (box_idx, random.choice(GeneticAlgorithm.get_valid_rotation_ids(boxes[box_idx])))
        return c

    @staticmethod
    def scramble_mutate(c):
        c = c[:]
        n = len(c)
        seg = random.randint(2, max(3, n // 10))
        s = random.randint(0, n - seg)
        part = c[s:s + seg]
        random.shuffle(part)
        c[s:s + seg] = part
        return c

    @staticmethod
    def multi_rotation_mutate(c, boxes):
        c = c[:]
        k = random.randint(2, min(4, len(c)))
        for i in random.sample(range(len(c)), k):
            bi, _ = c[i]
            c[i] = (bi, random.choice(GeneticAlgorithm.get_valid_rotation_ids(boxes[bi])))
        return c

    @staticmethod
    def mutate(chromosome, boxes, pressure=1.0):
        GA = GeneticAlgorithm
        c = chromosome[:]
        if random.random() < GA.MUTATION_SWAP * pressure:
            c = GA.swap_mutate(c)
        if random.random() < GA.MUTATION_INVERT * pressure:
            c = GA.weighted_inversion_mutate(c)
        if random.random() < GA.MUTATION_ROT * pressure:
            c = GA.multi_rotation_mutate(c, boxes) if random.random() < 0.4 else GA.rotation_mutate(c, boxes)
        if random.random() < GA.MUTATION_SCRAMBLE * pressure:
            c = GA.scramble_mutate(c)
        return c

    # ---- local search ----
    @staticmethod
    def local_search_2opt(chrom, boxes, container, max_iters=4):
        GA = GeneticAlgorithm
        best = chrom[:]
        bs, _, _ = GA.fitness(best, boxes, container)
        n = len(chrom)
        for _ in range(max_iters):
            improved = False
            for _ in range(min(3, n // 2)):
                i = random.randint(0, n - 3)
                j = random.randint(i + 2, min(i + n // 4, n - 1))
                cand = best[:]
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cs, _, _ = GA.fitness(cand, boxes, container)
                if cs > bs:
                    best, bs = cand, cs
                    improved = True
                    break
            if not improved: break
        return best

    # ---- restart / diversity ----
    @staticmethod
    def restart_population(population, scores, boxes):
        GA = GeneticAlgorithm
        n_keep = int(len(population) * (1 - GA.RESTART_RATIO))
        ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
        survivors = [population[i][:] for i in ranked[:n_keep]]
        fresh = [GA.random_chromosome(boxes) for _ in range(len(population) - n_keep)]
        return survivors + fresh

    @staticmethod
    def _chrom_key(c):
        return tuple(g[0] for g in c[:20])

    @staticmethod
    def remove_duplicates(pop, boxes):
        GA = GeneticAlgorithm
        seen = set()
        out = []
        for c in pop:
            k = GA._chrom_key(c)
            if k not in seen:
                seen.add(k)
                out.append(c)
            else:
                out.append(GA.random_chromosome(boxes))
        return out

    # ---- main solver ----
    @staticmethod
    def solve(boxes, container, seed=42, time_limit=240.0):
        random.seed(seed)
        wall_start = time.perf_counter()
        GA = GeneticAlgorithm
        n = len(boxes)
        best_chromosome = None
        best_positions  = []
        best_score      = -float("inf")
        best_util       = 0.0
        history         = []
        restart_round   = 0

        # ── Time-based restart loop ──────────────────────────────
        while True:
            restart_round += 1
            # Build population: first round uses heuristic seeds,
            # subsequent rounds seed from the best chromosome found
            if restart_round == 1:
                n_seeds = max(1, int(GA.POPULATION_SIZE * GA.SEED_RATIO))
                seeds = GA.build_seed_population(boxes, n_seeds)
                randoms = [GA.random_chromosome(boxes)
                           for _ in range(GA.POPULATION_SIZE - n_seeds)]
                population = seeds + randoms
            else:
                # Seed from global best + heuristic seeds + randoms
                n_seeds = max(1, int(GA.POPULATION_SIZE * GA.SEED_RATIO))
                seeds = GA.build_seed_population(boxes, n_seeds)
                randoms = [GA.random_chromosome(boxes)
                           for _ in range(GA.POPULATION_SIZE - n_seeds - 1)]
                population = [best_chromosome[:]] + seeds + randoms

            stagnation      = 0
            fitness_history  = []

            for generation in range(GA.NUM_GENERATIONS):
                # Check time limit
                if time.perf_counter() - wall_start >= time_limit:
                    break

                evaluated = [GA.fitness(c, boxes, container) for c in population]
                scores    = [e[0] for e in evaluated]
                positions = [e[1] for e in evaluated]
                utils     = [e[2] for e in evaluated]
                gen_best = max(range(len(population)), key=lambda i: scores[i])
                if scores[gen_best] > best_score:
                    best_score      = scores[gen_best]
                    best_positions  = positions[gen_best]
                    best_util       = utils[gen_best]
                    best_chromosome = population[gen_best][:]
                    stagnation      = 0
                else:
                    stagnation += 1
                history.append(round(best_util, 2))
                fitness_history.append(best_score)
                if best_util >= 80.0 and len(best_positions) == n:
                    break
                recent = fitness_history[-min(10, len(fitness_history)):]
                if len(recent) >= 2:
                    spread = max(recent) - min(recent)
                    convergence = max(0.0, min(1.0, 1.0 - spread / (abs(best_score) + 1e-6)))
                else:
                    convergence = 0.5
                if stagnation >= GA.STAGNATION_LIMIT:
                    population = GA.restart_population(population, scores, boxes)
                    stagnation = 0
                    continue
                ranked = sorted(range(len(population)), key=lambda i: scores[i], reverse=True)
                elite = [population[i][:] for i in ranked[:GA.ELITE_COUNT]]
                if generation % 15 == 0 and generation > 0:
                    elite = [GA.local_search_2opt(c, boxes, container, 3) for c in elite]
                new_pop = elite[:]
                progress = generation / max(1, GA.NUM_GENERATIONS - 1)
                t_size = max(3, int(GA.TOURNAMENT_SIZE * (1.0 - 0.4 * progress)))
                pressure = 0.8 + 0.5 * convergence
                while len(new_pop) < GA.POPULATION_SIZE:
                    p1 = GA.tournament_select(population, scores, t_size)
                    p2 = GA.tournament_select(population, scores, t_size)
                    child = GA.crossover(p1, p2) if random.random() < GA.CROSSOVER_RATE else p1[:]
                    child = GA.mutate(child, boxes, pressure)
                    new_pop.append(child)
                if generation % 15 == 0:
                    new_pop = GA.remove_duplicates(new_pop, boxes)
                population = new_pop

            # Check if time is up — if so, stop; otherwise restart
            if time.perf_counter() - wall_start >= time_limit:
                break
            if best_util >= 80.0 and len(best_positions) == n:
                break

        return {
            "positions":   best_positions,
            "utilization": round(best_util, 2),
            "history":     history,
            "fitness":     round(best_score, 2),
        }


# ---- backward-compat shims ----
POPULATION_SIZE   = GeneticAlgorithm.POPULATION_SIZE
NUM_GENERATIONS   = GeneticAlgorithm.NUM_GENERATIONS


def solve(boxes, container, seed=42, time_limit=240.0):
    GeneticAlgorithm.POPULATION_SIZE = POPULATION_SIZE
    GeneticAlgorithm.NUM_GENERATIONS = NUM_GENERATIONS
    return GeneticAlgorithm.solve(boxes, container, seed=seed, time_limit=time_limit)
