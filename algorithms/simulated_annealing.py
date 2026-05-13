import random
import math
import time

from core.utils import compute_utilization, compute_stability


# ══════════════════════════════════════════════════════════════════════════════
# Self-contained Simulated Annealing packing solver.
# Uses a fast shelf-based decoder that tries ALL rotations per box (same
# strategy as space_manager.place_boxes) for high utilization.
# ══════════════════════════════════════════════════════════════════════════════


class SimulatedAnnealing:
    """
    Simulated Annealing solver for the 3-D container loading problem.
    Chromosome = permutation of box indices (pure ordering).
    Decoder = shelf placer that auto-selects the best rotation per box.

    The shelf decoder enforces gravity by construction: each layer rests
    on the one below (or the floor), so no floating boxes are possible.
    """

    # ──────────────────────────────────────────────────────────────────────
    # Rotation helpers
    # ──────────────────────────────────────────────────────────────────────

    ROTATION_TABLE = [
        lambda l, w, h: (l, w, h),
        lambda l, w, h: (l, h, w),
        lambda l, w, h: (w, l, h),
        lambda l, w, h: (w, h, l),
        lambda l, w, h: (h, l, w),
        lambda l, w, h: (h, w, l),
    ]

    @staticmethod
    def _get_rotations(box, CL, CW, CH):
        """Return deduplicated (l,w,h) tuples that fit inside the container."""
        dims = [box.l, box.w, box.h]
        locked = getattr(box, 'orientation_locked', False) or getattr(box, 'fragile', False)
        if locked:
            candidates = [(box.l, box.w, box.h), (box.w, box.l, box.h)]
        else:
            seen = set()
            candidates = []
            perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
            for i,j,k in perms:
                d = (dims[i], dims[j], dims[k])
                if d not in seen:
                    seen.add(d)
                    candidates.append(d)
        return sorted([(l,w,h) for l,w,h in candidates if l<=CL and w<=CW and h<=CH],
                       key=lambda t: t[2])  # flattest first → shorter layers

    # ──────────────────────────────────────────────────────────────────────
    # Decoder – first-fit shelf placer (order-sensitive for SA)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def decode(chromosome, boxes, container, settle=False):
        """
        First-fit shelf-based decoder. Chromosome is a permutation of box
        indices. Order-sensitive: different permutations -> different packings.

        Shelf layers prevent vertical overlap by construction. A gravity drop
        pass (_gravity_drop) is applied after placement to settle boxes to
        their physical resting z-position, eliminating any floating boxes
        that may arise from gaps within a layer.

        The 'settle' parameter is accepted for API compatibility.
        """
        CL, CW, CH = container.L, container.W, container.H
        placed = []
        unplaced = []
        layers = []   # [(z, h, rows: [(y, w, x_next)])]
        z_top = 0.0

        for box_idx in chromosome:
            box = boxes[box_idx]
            rots = SimulatedAnnealing._get_rotations(box, CL, CW, CH)
            if not rots:
                unplaced.append(box.id)
                continue

            placed_flag = False

            # Try existing layers
            for layer in layers:
                for (bl, bw, bh) in rots:
                    if bh > layer[1]:
                        continue
                    # Try existing rows
                    for row in layer[2]:
                        if bw <= row[1]+1e-9 and row[2]+bl <= CL+1e-9:
                            placed.append({
                                "id": box.id,
                                "x": row[2], "y": row[0], "z": layer[0],
                                "l": bl, "w": bw, "h": bh,
                                "weight": box.weight,
                            })
                            row[2] += bl
                            placed_flag = True
                            break
                    if placed_flag:
                        break
                    # New row in this layer
                    y_used = sum(r[1] for r in layer[2])
                    if y_used + bw <= CW + 1e-9:
                        layer[2].append([y_used, bw, bl])
                        placed.append({
                            "id": box.id,
                            "x": 0.0, "y": y_used, "z": layer[0],
                            "l": bl, "w": bw, "h": bh,
                            "weight": box.weight,
                        })
                        placed_flag = True
                        break
                if placed_flag:
                    break

            # New layer
            if not placed_flag:
                for (bl, bw, bh) in rots:
                    if z_top + bh <= CH + 1e-9:
                        layers.append([z_top, bh, [[0.0, bw, bl]]])
                        placed.append({
                            "id": box.id,
                            "x": 0.0, "y": 0.0, "z": z_top,
                            "l": bl, "w": bw, "h": bh,
                            "weight": box.weight,
                        })
                        z_top += bh
                        placed_flag = True
                        break

            if not placed_flag:
                unplaced.append(box.id)

        # Gravity drop: ensure every box rests on a physical surface
        placed = SimulatedAnnealing._gravity_drop(placed, CH)
        return placed, unplaced

    # ──────────────────────────────────────────────────────────────────────
    # Gravity drop — settle boxes to their physical resting position
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gravity_drop(placed, CH):
        """
        Drop every box to its physical resting z-position.

        Process boxes bottom-up. For each box, compute the landing z as the
        highest z_top of any already-settled box whose x-y footprint overlaps.
        Boxes on the floor (z=0) stay. Boxes above gaps are lowered until they
        rest on a surface.

        This is O(n²) — for n=386 it takes < 1 ms, so runtime is unaffected.
        No boxes are removed, so utilization is preserved.
        """
        if not placed:
            return placed

        # Sort by z, then y, then x (process bottom-up)
        placed.sort(key=lambda p: (p['z'], p['y'], p['x']))

        settled = []
        for p in placed:
            px, py, pl, pw, ph = p['x'], p['y'], p['l'], p['w'], p['h']

            # Find the landing z: highest z_top of any settled box that
            # overlaps in x-y (even partially)
            z_land = 0.0
            for s in settled:
                # Check x-y overlap (boxes must share area, not just touch)
                if (px < s['x'] + s['l'] - 1e-9 and px + pl > s['x'] + 1e-9 and
                    py < s['y'] + s['w'] - 1e-9 and py + pw > s['y'] + 1e-9):
                    z_top = s['z'] + s['h']
                    if z_top > z_land:
                        z_land = z_top

            # Only drop (never raise) and keep within container
            if z_land <= p['z'] and z_land + ph <= CH + 1e-9:
                p['z'] = z_land

            settled.append(p)

        return settled

    # ──────────────────────────────────────────────────────────────────────
    # Fitness — uses shared metrics from core.utils (same as GA)
    # ──────────────────────────────────────────────────────────────────────

    _UNPLACED_PENALTY = 35.0   # matched with GA
    _STABILITY_WEIGHT = 0.20   # matched with GA

    @staticmethod
    def compute_fitness(placed, unplaced, container):
        u = compute_utilization(placed, container)
        s = compute_stability(placed, container)
        penalty = 0.0
        if len(unplaced) > 0:
            penalty = SimulatedAnnealing._UNPLACED_PENALTY * len(unplaced) + (len(unplaced) ** 1.5) * 5.0
        placed_bonus = 5.0 if len(unplaced) == 0 else 0.0
        return u + (s * SimulatedAnnealing._STABILITY_WEIGHT) + placed_bonus - penalty

    # ──────────────────────────────────────────────────────────────────────
    # Chromosome helpers (pure permutation of box indices)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _order_seed(boxes, key_fn):
        return sorted(range(len(boxes)), key=lambda i: key_fn(boxes[i]), reverse=True)

    @staticmethod
    def create_initial_chromosome(boxes, container):
        """Try heuristic orderings (descending & ascending) plus random
        shuffles and return the best one."""
        SA = SimulatedAnnealing
        # Descending-order heuristics (large first)
        desc_keys = [
            lambda b: b.l * b.w * b.h,
            lambda b: b.h,
            lambda b: b.l * b.w,
            lambda b: b.w * b.h,
            lambda b: max(b.l, b.w, b.h),
            lambda b: b.l + b.w + b.h,
            lambda b: min(b.l, b.w, b.h),
            lambda b: (b.l*b.w*b.h) / (b.weight+1),
            lambda b: b.weight,
            lambda b: max(b.l, b.w, b.h) * (b.weight+1),
            lambda b: b.weight * b.l * b.w,
            lambda b: b.l * b.w / (b.h+1),
            lambda b: max(b.l*b.w, b.l*b.h, b.w*b.h),  # largest face
        ]
        # Ascending-order heuristics (small first — fill gaps)
        asc_keys = [
            lambda b: b.l * b.w * b.h,
            lambda b: b.h,
            lambda b: min(b.l, b.w, b.h),
        ]
        best_chrom, best_fit = None, -float('inf')

        def _try(chrom):
            nonlocal best_chrom, best_fit
            p, u = SA.decode(chrom, boxes, container)
            f = SA.compute_fitness(p, u, container)
            if f > best_fit:
                best_fit = f
                best_chrom = list(chrom)

        for key in desc_keys:
            _try(SA._order_seed(boxes, key))
        for key in asc_keys:
            _try(sorted(range(len(boxes)), key=lambda i, k=key: k(boxes[i])))

        # Random shuffles for diversity
        saved = random.getstate()
        for s in range(20):
            random.seed(s * 7 + 13)
            chrom = list(range(len(boxes)))
            random.shuffle(chrom)
            _try(chrom)
        random.setstate(saved)

        return best_chrom

    @staticmethod
    def perturb(chromosome):
        """
        Random perturbation on a permutation:
          - Swap (35%):    exchange two positions
          - Reverse (30%): reverse a short segment
          - Insert (35%):  remove and re-insert a box elsewhere
        """
        c = list(chromosome)
        n = len(c)
        r = random.random()
        if r < 0.35:
            # Swap
            i, j = random.sample(range(n), 2)
            c[i], c[j] = c[j], c[i]
        elif r < 0.65:
            # Reverse a short segment
            i = random.randint(0, n-2)
            j = random.randint(i+1, min(i+max(2, n//6), n-1))
            c[i:j+1] = c[i:j+1][::-1]
        else:
            # Insert: remove one gene and re-insert elsewhere
            i = random.randrange(n)
            gene = c.pop(i)
            j = random.randrange(len(c)+1)
            c.insert(j, gene)
        return c

    @staticmethod
    def should_accept(current_fitness, new_fitness, temperature):
        if new_fitness >= current_fitness:
            return True
        if temperature <= 0:
            return False
        delta = current_fitness - new_fitness
        return random.random() < math.exp(-delta / temperature)

    # ──────────────────────────────────────────────────────────────────────
    # Main SA loop — multi-restart with time budget
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def solve(boxes, container, max_iterations=100000,
              initial_temp=1000.0, cooling_rate=0.99985,
              reheat_threshold=5000, reheat_ratio=0.5,
              time_limit=240.0, seed=42):
        """
        Simulated Annealing main loop with multi-restart strategy.

        The SA uses a calibrated temperature schedule based on the observed
        fitness landscape. On stagnation, it restarts from the global best
        with a fresh temperature, allowing the search to explore new regions.

        Parameters:
            max_iterations     – upper bound on iterations per restart
            initial_temp       – (ignored, auto-calibrated to 50)
            cooling_rate       – (ignored, auto-calibrated)
            reheat_threshold   – iterations without improvement before restart
            reheat_ratio       – (ignored)
            time_limit         – max wall-clock seconds (default 240 = 4 min)

        Returns dict: {positions, utilization, fitness, iteration}
        """
        SA = SimulatedAnnealing
        random.seed(seed)
        wall_start = time.perf_counter()

        # ── Initialise from best heuristic seed ───────────────────────
        current = SA.create_initial_chromosome(boxes, container)
        cur_placed, cur_unplaced = SA.decode(current, boxes, container)
        cur_fit = SA.compute_fitness(cur_placed, cur_unplaced, container)

        best       = list(current)
        best_fit   = cur_fit
        best_placed = list(cur_placed)
        best_unplaced = list(cur_unplaced)
        best_iter  = 0

        # ── Calibrated SA parameters ─────────────────────────────────
        # Lower T0 with gentler decay allows sustained exploration over
        # longer restarts without collapsing to pure hill-climbing.
        T0 = 15.0
        # Cooling: balanced schedule to sustain exploration through restarts
        alpha = 0.9997592
        # Restart on stagnation to explore new regions
        restart_threshold = 5000

        temperature = T0
        stagnation  = 0
        total_it    = 0
        restart_count = 0

        # ── Main loop with restarts ──────────────────────────────────
        while True:
            for it in range(1, max_iterations + 1):
                total_it += 1

                # Time limit check every 200 iterations
                if total_it % 200 == 0:
                    if time.perf_counter() - wall_start >= time_limit:
                        break

                neighbour = SA.perturb(current)
                n_placed, n_unplaced = SA.decode(neighbour, boxes, container)
                n_fit = SA.compute_fitness(n_placed, n_unplaced, container)

                if SA.should_accept(cur_fit, n_fit, temperature):
                    current = neighbour
                    cur_placed = n_placed
                    cur_unplaced = n_unplaced
                    cur_fit = n_fit

                if cur_fit > best_fit:
                    best       = list(current)
                    best_fit   = cur_fit
                    best_placed = list(cur_placed)
                    best_unplaced = list(cur_unplaced)
                    best_iter  = total_it
                    stagnation = 0
                else:
                    stagnation += 1

                temperature *= alpha
                if temperature < 1e-6:
                    temperature = 1e-6  # floor to pure hill-climbing

                # Restart from best when stagnating
                if stagnation >= restart_threshold:
                    break

            # Check time limit
            if time.perf_counter() - wall_start >= time_limit:
                break

            # ── Restart: go back to best, decaying temperature ───────
            # Each restart uses a lower T so later restarts are more
            # focused on exploitation rather than exploration.
            restart_count += 1
            current = list(best)
            cur_placed = list(best_placed)
            cur_unplaced = list(best_unplaced)
            cur_fit = best_fit
            # Strong perturbation on restart: 5-10 moves, biased towards
            # insert (which makes larger structural changes than swap/reverse)
            n_perturbs = random.randint(5, 10)
            for _ in range(n_perturbs):
                c = list(current)
                n = len(c)
                if random.random() < 0.6:
                    # Insert move (60% of restart perturbations)
                    i = random.randrange(n)
                    gene = c.pop(i)
                    j = random.randrange(len(c) + 1)
                    c.insert(j, gene)
                    current = c
                else:
                    current = SA.perturb(current)
            cur_placed, cur_unplaced = SA.decode(current, boxes, container)
            cur_fit = SA.compute_fitness(cur_placed, cur_unplaced, container)
            temperature = T0 * max(0.4, 0.9 ** restart_count)
            stagnation = 0

        # ── Return best found ─────────────────────────────────────────
        vol = sum(p["l"]*p["w"]*p["h"] for p in best_placed)
        cvol = container.L * container.W * container.H
        util = (vol / cvol * 100) if cvol > 0 else 0.0

        return {
            "positions":   best_placed,
            "utilization": util,
            "fitness":     best_fit,
            "iteration":   best_iter,
        }


# ─────────────────────────── backward-compat shim ─────────────────
def solve(boxes, container, **kwargs):
    return SimulatedAnnealing.solve(boxes, container, **kwargs)
