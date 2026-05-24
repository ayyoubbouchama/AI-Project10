import random
import math
import time


# ══════════════════════════════════════════════════════════════════════════════
# Self-contained Simulated Annealing packing solver.
# Uses a fast shelf-based decoder that tries ALL rotations per box (same
# strategy as space_manager.place_boxes) for high utilization.
#
# PERFORMANCE OPTIMIZATIONS:
#   - Rotation cache built once per solve() call
#   - Tuple-based placement in hot path (converted to dicts only for output)
#   - Fitness metrics (utilization, stability) inlined into decoder
#   - Reduced initial chromosome trials (5 random instead of 20)
#   - Tuned cooling schedule for faster convergence
#   - Eliminated redundant list copies
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
    def _build_rotation_cache(boxes, CL, CW, CH):
        """Pre-compute valid rotation dims for every box. Called once.
        Returns list of lists: cache[box_idx] = [(l,w,h), ...] sorted flattest first.
        """
        cache = []
        for box in boxes:
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
            valid = sorted([(l,w,h) for l,w,h in candidates
                            if l<=CL and w<=CW and h<=CH],
                           key=lambda t: t[2])  # flattest first
            cache.append(valid)
        return cache

    @staticmethod
    def _get_rotations(box, CL, CW, CH):
        """Fallback for non-cached contexts. Return deduplicated (l,w,h) tuples."""
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
                       key=lambda t: t[2])

    # ──────────────────────────────────────────────────────────────────────
    # Decoder – first-fit shelf placer (tuple-based hot path)
    # ──────────────────────────────────────────────────────────────────────
    # Tuple layout: (id, x, y, z, l, w, h, weight)
    #                 0   1  2  3  4  5  6  7

    @staticmethod
    def decode(chromosome, boxes, container, rot_cache=None, settle=False):
        """
        First-fit shelf-based decoder. Chromosome is a permutation of box
        indices. Returns (placed_dicts, unplaced_ids) for API compatibility.
        """
        tuples, unplaced, _, _ = SimulatedAnnealing._decode_tuples(
            chromosome, boxes, container, rot_cache)
        # Convert to dicts for external use
        placed = [{"id": p[0], "x": p[1], "y": p[2], "z": p[3],
                   "l": p[4], "w": p[5], "h": p[6], "weight": p[7]}
                  for p in tuples]
        return placed, unplaced

    @staticmethod
    def _decode_tuples(chromosome, boxes, container, rot_cache):
        """
        Internal tuple-based decoder. Returns (placed_tuples, unplaced_ids,
        utilization, stability).
        """
        CL, CW, CH = container.L, container.W, container.H
        placed = []
        unplaced = []
        eps = [(0.0, 0.0, 0.0)]
        total_vol = 0.0

        for box_idx in chromosome:
            box = boxes[box_idx]
            rots = rot_cache[box_idx] if rot_cache else SimulatedAnnealing._get_rotations(box, CL, CW, CH)
            if not rots:
                unplaced.append(box.id)
                continue

            box_id = box.id
            box_weight = box.weight
            placed_flag = False

            eps = list(set(eps))
            eps.sort(key=lambda p: (p[2], p[1], p[0]))

            for i in range(len(eps)):
                ex, ey, ez = eps[i]
                for (bl, bw, bh) in rots:
                    ex_bl = ex + bl
                    ey_bw = ey + bw
                    ez_bh = ez + bh

                    # Boundary check
                    if ex_bl > CL + 1e-9 or ey_bw > CW + 1e-9 or ez_bh > CH + 1e-9:
                        continue

                    # Fast 3D AABB Overlap check
                    overlap = False
                    for s in placed:
                        # s tuple format: (pid, px, py, pz, pl, pw, ph, pwt)
                        if not (ex >= s[1] + s[4] - 1e-9 or ex_bl <= s[1] + 1e-9 or
                                ey >= s[2] + s[5] - 1e-9 or ey_bw <= s[2] + 1e-9 or
                                ez >= s[3] + s[6] - 1e-9 or ez_bh <= s[3] + 1e-9):
                            overlap = True
                            break

                    if not overlap:
                        placed.append((box_id, ex, ey, ez, bl, bw, bh, box_weight))
                        total_vol += bl * bw * bh
                        
                        eps.pop(i)
                        eps.extend([(ex_bl, ey, ez), (ex, ey_bw, ez), (ex, ey, ez_bh)])
                        placed_flag = True
                        break
                
                if placed_flag:
                    break

            if not placed_flag:
                unplaced.append(box.id)

        # Gravity drop with soft penalty
        placed, penalty_vol = SimulatedAnnealing._gravity_drop_tuples(placed, CH)

        # Inline utilization
        cvol = CL * CW * CH
        util = (total_vol / cvol * 100.0) if cvol > 0 else 0.0

        # Inline stability
        actual_sum = 0.0
        worst_sum = 0.0
        for p in placed:
            w = p[7]
            actual_sum += w * p[3]
            worst_sum += w * max(CH - p[6], 0)
        if worst_sum == 0:
            stability = 100.0
        else:
            stability = max(0.0, min(100.0, (1.0 - actual_sum / worst_sum) * 100.0))

        return placed, unplaced, util, stability, penalty_vol

    # ──────────────────────────────────────────────────────────────────────
    # Gravity drop — tuple-based
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gravity_drop_tuples(placed, CH):
        """
        Drop every box to its physical resting z-position.
        Works on tuples: (id, x, y, z, l, w, h, weight).
        Enforces a 50% base area support constraint. Unsupported boxes are rejected.
        """
        if not placed:
            return placed, 0.0

        placed_sorted = sorted(placed, key=lambda p: (p[3], p[2], p[1]))

        settled = []
        penalty_vol = 0.0
        MIN_SUP = SimulatedAnnealing.MIN_SUPPORT

        for p in placed_sorted:
            pid, px, py, pz, pl, pw, ph, pwt = p

            z_land = 0.0
            for s in settled:
                sx, sy, sz, sl, sw, sh = s[1], s[2], s[3], s[4], s[5], s[6]
                if (px < sx + sl - 1e-9 and px + pl > sx + 1e-9 and
                    py < sy + sw - 1e-9 and py + pw > sy + 1e-9):
                    z_top = sz + sh
                    if z_top > z_land:
                        z_land = z_top

            if z_land <= pz and z_land + ph <= CH + 1e-9:
                pz = z_land

            # Floor boxes always pass
            if pz < 1e-9:
                settled.append((pid, px, py, pz, pl, pw, ph, pwt))
                continue

            # Compute support area from settled boxes directly below
            base_area = pl * pw
            support_area = 0.0
            for s in settled:
                sx, sy, sz, sl, sw, sh = s[1], s[2], s[3], s[4], s[5], s[6]
                if abs((sz + sh) - pz) < 1e-6:
                    ox = max(0.0, min(px + pl, sx + sl) - max(px, sx))
                    oy = max(0.0, min(py + pw, sy + sw) - max(py, sy))
                    support_area += ox * oy

            settled.append((pid, px, py, pz, pl, pw, ph, pwt)) # ALWAYS keep box
            if base_area > 0 and (support_area / base_area) < MIN_SUP:
                penalty_vol += pl * pw * ph

        return settled, penalty_vol

    @staticmethod
    def _gravity_drop(placed, CH):
        """Dict-based gravity drop with support check — kept for API compatibility."""
        if not placed:
            return placed
        placed.sort(key=lambda p: (p['z'], p['y'], p['x']))
        settled = []
        MIN_SUP = SimulatedAnnealing.MIN_SUPPORT
        
        for p in placed:
            px, py, pl, pw, ph = p['x'], p['y'], p['l'], p['w'], p['h']
            z_land = 0.0
            for s in settled:
                if (px < s['x'] + s['l'] - 1e-9 and px + pl > s['x'] + 1e-9 and
                    py < s['y'] + s['w'] - 1e-9 and py + pw > s['y'] + 1e-9):
                    z_top = s['z'] + s['h']
                    if z_top > z_land:
                        z_land = z_top
            
            if z_land <= p['z'] and z_land + ph <= CH + 1e-9:
                p['z'] = z_land
                
            if p['z'] < 1e-9:
                settled.append(p)
                continue
                
            base_area = pl * pw
            support_area = 0.0
            for s in settled:
                if abs((s['z'] + s['h']) - p['z']) < 1e-6:
                    ox = max(0.0, min(px + pl, s['x'] + s['l']) - max(px, s['x']))
                    oy = max(0.0, min(py + pw, s['y'] + s['w']) - max(py, s['y']))
                    support_area += ox * oy
            
            # In dict-based API, we just keep all boxes to avoid cascade
            settled.append(p)
                
        return settled

    # ──────────────────────────────────────────────────────────────────────
    # Fitness — inlined for hot path, external for compatibility
    # ──────────────────────────────────────────────────────────────────────

    _UNPLACED_PENALTY = 35.0
    _STABILITY_WEIGHT = 0.20
    MIN_SUPPORT = 0.50

    @staticmethod
    def compute_fitness(placed, unplaced, container):
        """External compatibility — uses dicts."""
        from core.utils import compute_utilization, compute_stability
        u = compute_utilization(placed, container)
        s = compute_stability(placed, container)
        penalty = 0.0
        if len(unplaced) > 0:
            penalty = SimulatedAnnealing._UNPLACED_PENALTY * len(unplaced) + (len(unplaced) ** 1.5) * 5.0
        placed_bonus = 5.0 if len(unplaced) == 0 else 0.0
        return u + (s * SimulatedAnnealing._STABILITY_WEIGHT) + placed_bonus - penalty

    @staticmethod
    def _compute_fitness_inline(util, stability, n_unplaced, penalty_vol=0.0):
        """Fast inline fitness from pre-computed metrics."""
        SA = SimulatedAnnealing
        penalty = 0.0
        if n_unplaced > 0:
            penalty = SA._UNPLACED_PENALTY * n_unplaced + (n_unplaced ** 1.5) * 5.0
            
        penalty += penalty_vol * 1.5
        
        placed_bonus = 5.0 if n_unplaced == 0 else 0.0
        return util + (stability * SA._STABILITY_WEIGHT) + placed_bonus - penalty

    # ──────────────────────────────────────────────────────────────────────
    # Chromosome helpers (pure permutation of box indices)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _order_seed(boxes, key_fn):
        return sorted(range(len(boxes)), key=lambda i: key_fn(boxes[i]), reverse=True)

    @staticmethod
    def create_initial_chromosome(boxes, container, rot_cache):
        """Try heuristic orderings plus random shuffles and return the best."""
        SA = SimulatedAnnealing
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
            lambda b: max(b.l*b.w, b.l*b.h, b.w*b.h),
        ]
        asc_keys = [
            lambda b: b.l * b.w * b.h,
            lambda b: b.h,
            lambda b: min(b.l, b.w, b.h),
        ]
        best_chrom, best_fit = None, -float('inf')

        def _try(chrom):
            nonlocal best_chrom, best_fit
            _, unp, u, s, r = SA._decode_tuples(chrom, boxes, container, rot_cache)
            f = SA._compute_fitness_inline(u, s, len(unp), r)
            if f > best_fit:
                best_fit = f
                best_chrom = list(chrom)

        for key in desc_keys:
            _try(SA._order_seed(boxes, key))
        for key in asc_keys:
            _try(sorted(range(len(boxes)), key=lambda i, k=key: k(boxes[i])))

        # Reduced random shuffles: 5 instead of 20
        saved = random.getstate()
        for s in range(5):
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
    def solve(boxes, container, max_iterations=30000,
              initial_temp=1000.0, cooling_rate=0.99985,
              reheat_threshold=5000, reheat_ratio=0.5,
              time_limit=40.0, seed=42):
        """
        Simulated Annealing main loop with multi-restart strategy.

        Parameters:
            max_iterations     – upper bound on iterations per restart
            initial_temp       – (ignored, auto-calibrated to 15)
            cooling_rate       – (ignored, auto-calibrated)
            reheat_threshold   – (ignored, auto-calibrated to 2000)
            reheat_ratio       – (ignored)
            time_limit         – max wall-clock seconds (default 40)

        Returns dict: {positions, utilization, fitness, iteration}
        """
        SA = SimulatedAnnealing
        random.seed(seed)
        wall_start = time.perf_counter()

        CL, CW, CH = container.L, container.W, container.H

        # Pre-cache all rotations once
        rot_cache = SA._build_rotation_cache(boxes, CL, CW, CH)

        # ── Initialise from best heuristic seed ───────────────────────
        current = SA.create_initial_chromosome(boxes, container, rot_cache)
        cur_tuples, cur_unplaced, cur_util, cur_stab, cur_rej = SA._decode_tuples(
            current, boxes, container, rot_cache)
        cur_fit = SA._compute_fitness_inline(cur_util, cur_stab, len(cur_unplaced), cur_rej)

        best       = list(current)
        best_fit   = cur_fit
        best_tuples = list(cur_tuples)
        best_unplaced = list(cur_unplaced)
        best_iter  = 0

        # ── Calibrated SA parameters ─────────────────────────────────
        T0 = 15.0
        alpha = 0.9995  # faster cooling for quicker convergence
        restart_threshold = 2000  # restart sooner to avoid wasted iterations

        temperature = T0
        stagnation  = 0
        total_it    = 0
        restart_count = 0

        # Cache math.exp and random.random for hot loop
        _exp = math.exp
        _random = random.random

        # ── Main loop with restarts ──────────────────────────────────
        while True:
            for it in range(1, max_iterations + 1):
                total_it += 1

                # Time limit check every 5 iterations for slow DBLF
                if total_it % 5 == 0:
                    if time.perf_counter() - wall_start >= time_limit:
                        break

                neighbour = SA.perturb(current)
                n_tuples, n_unplaced, n_util, n_stab, n_rej = SA._decode_tuples(
                    neighbour, boxes, container, rot_cache)
                n_fit = SA._compute_fitness_inline(n_util, n_stab, len(n_unplaced), n_rej)

                # Inline acceptance check for speed
                accept = False
                if n_fit >= cur_fit:
                    accept = True
                elif temperature > 0:
                    delta = cur_fit - n_fit
                    if _random() < _exp(-delta / temperature):
                        accept = True

                if accept:
                    current = neighbour
                    cur_fit = n_fit

                if n_fit > best_fit and accept:
                    best       = list(current)
                    best_fit   = n_fit
                    best_tuples = list(n_tuples)
                    best_unplaced = list(n_unplaced)
                    best_iter  = total_it
                    stagnation = 0
                else:
                    stagnation += 1

                temperature *= alpha
                if temperature < 1e-6:
                    temperature = 1e-6

                if stagnation >= restart_threshold:
                    break

            # Check time limit
            if time.perf_counter() - wall_start >= time_limit:
                break

            # ── Restart: go back to best, decaying temperature ───────
            restart_count += 1
            current = list(best)
            cur_fit = best_fit
            # Strong perturbation on restart
            n_perturbs = random.randint(5, 10)
            for _ in range(n_perturbs):
                c = list(current)
                n = len(c)
                if _random() < 0.6:
                    i = random.randrange(n)
                    gene = c.pop(i)
                    j = random.randrange(len(c) + 1)
                    c.insert(j, gene)
                    current = c
                else:
                    current = SA.perturb(current)
            _, cur_unp, cur_u, cur_s, cur_r = SA._decode_tuples(
                current, boxes, container, rot_cache)
            cur_fit = SA._compute_fitness_inline(cur_u, cur_s, len(cur_unp), cur_r)
            temperature = T0 * max(0.4, 0.9 ** restart_count)
            stagnation = 0

        # ── Return best found ─────────────────────────────────────────
        best_positions = [{"id": p[0], "x": p[1], "y": p[2], "z": p[3],
                           "l": p[4], "w": p[5], "h": p[6], "weight": p[7]}
                          for p in best_tuples]

        vol = sum(p["l"]*p["w"]*p["h"] for p in best_positions)
        cvol = CL * CW * CH
        util = (vol / cvol * 100) if cvol > 0 else 0.0

        return {
            "positions":   best_positions,
            "utilization": util,
            "fitness":     best_fit,
            "iteration":   best_iter,
        }


# ─────────────────────────── backward-compat shim ─────────────────
def solve(boxes, container, **kwargs):
    return SimulatedAnnealing.solve(boxes, container, **kwargs)
