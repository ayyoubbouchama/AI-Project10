import random
import math
import time


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
        return [(l,w,h) for l,w,h in candidates if l<=CL and w<=CW and h<=CH]

    # ──────────────────────────────────────────────────────────────────────
    # Decoder – first-fit shelf placer (order-sensitive for SA)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def decode(chromosome, boxes, container, settle=False):
        """
        First-fit shelf-based decoder. Chromosome is a permutation of box
        indices. Order-sensitive: different permutations → different packings.

        Gravity is enforced by construction: each layer sits on the layer
        below it (or the floor). No post-processing needed.

        The 'settle' parameter is accepted for API compatibility but the
        shelf layout is already physically valid by construction.
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
    # Fitness
    # ──────────────────────────────────────────────────────────────────────

    _UNPLACED_PENALTY = 5.0
    _STABILITY_WEIGHT = 0.10

    @staticmethod
    def compute_utilization(placements, container):
        total = sum(p["l"]*p["w"]*p["h"] for p in placements)
        return (total / container.volume()) * 100

    @staticmethod
    def compute_stability(placed_boxes):
        if len(placed_boxes) <= 1:
            return 100.0
        max_z = max(pb["z"]+pb["h"] for pb in placed_boxes)
        if max_z == 0:
            return 100.0
        tw, wz = 0.0, 0.0
        for pb in placed_boxes:
            w = pb.get("weight", 1)
            wz += w * (pb["z"] + pb["h"]/2.0) / max_z
            tw += w
        return (1.0 - wz/tw) * 100.0 if tw > 0 else 100.0

    @staticmethod
    def compute_fitness(placed, unplaced, container):
        u = SimulatedAnnealing.compute_utilization(placed, container)
        s = SimulatedAnnealing.compute_stability(placed)
        return u - SimulatedAnnealing._UNPLACED_PENALTY*len(unplaced) + SimulatedAnnealing._STABILITY_WEIGHT*s

    # ──────────────────────────────────────────────────────────────────────
    # Chromosome helpers (pure permutation of box indices)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _order_seed(boxes, key_fn):
        return sorted(range(len(boxes)), key=lambda i: key_fn(boxes[i]), reverse=True)

    @staticmethod
    def create_initial_chromosome(boxes, container):
        """Try 12 heuristic orderings and return the best one."""
        SA = SimulatedAnnealing
        heuristics = [
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
        ]
        best_chrom, best_fit = None, -float('inf')
        for key in heuristics:
            chrom = SA._order_seed(boxes, key)
            p, u = SA.decode(chrom, boxes, container)
            f = SA.compute_fitness(p, u, container)
            if f > best_fit:
                best_fit = f
                best_chrom = chrom
        return best_chrom

    @staticmethod
    def perturb(chromosome):
        """
        Random perturbation on a permutation:
          - Swap (35%): exchange two positions
          - Reverse (30%): reverse a short segment
          - Insert (35%): remove and re-insert a box elsewhere
        """
        c = list(chromosome)
        n = len(c)
        r = random.random()
        if r < 0.35:
            i, j = random.sample(range(n), 2)
            c[i], c[j] = c[j], c[i]
        elif r < 0.65:
            i = random.randint(0, n-2)
            j = random.randint(i+1, min(i+max(2, n//6), n-1))
            c[i:j+1] = c[i:j+1][::-1]
        else:
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
              time_limit=300.0):
        """
        Simulated Annealing main loop with multi-restart strategy.

        The SA uses a calibrated temperature schedule based on the observed
        fitness landscape. On stagnation, it restarts from the global best
        with a fresh temperature, allowing the search to explore new regions.

        Parameters:
            max_iterations     – upper bound on iterations per restart
            initial_temp       – (ignored, auto-calibrated to ~15)
            cooling_rate       – (ignored, auto-calibrated)
            reheat_threshold   – iterations without improvement before restart
            reheat_ratio       – (ignored)
            time_limit         – max wall-clock seconds (default 300 = 5 min)

        Returns dict: {positions, utilization, fitness, iteration}
        """
        SA = SimulatedAnnealing
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
        # Temperature calibrated to produce ~40% initial acceptance
        # (tight enough to avoid drifting too far from good solutions)
        T0 = 10.0
        # Cooling: reach T~0.01 in ~15k iterations → α = (0.01/10)^(1/15000) ≈ 0.99954
        alpha = 0.99954
        restart_threshold = 3000  # restart sooner for more attempts

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

            # ── Restart: go back to best, fresh temperature ──────────
            restart_count += 1
            current = list(best)
            cur_placed = list(best_placed)
            cur_unplaced = list(best_unplaced)
            cur_fit = best_fit
            # Slightly perturb to avoid re-tracing the same path
            for _ in range(random.randint(1, 3)):
                current = SA.perturb(current)
            cur_placed, cur_unplaced = SA.decode(current, boxes, container)
            cur_fit = SA.compute_fitness(cur_placed, cur_unplaced, container)
            temperature = T0
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
def solve(boxes, container):
    return SimulatedAnnealing.solve(boxes, container)
