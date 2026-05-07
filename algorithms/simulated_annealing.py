import random
import math


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
    # Gravity settling  (ported from GA)
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _gravity_settle(placed, CL, CW, CH):
        if not placed:
            return placed
        MIN_SUPPORT = 0.50
        placed.sort(key=lambda p: (p["z"], p["y"], p["x"]))
        settled = []
        for p in placed:
            px, py = p["x"], p["y"]
            pl, pw, ph = p["l"], p["w"], p["h"]
            base_area = pl * pw
            z_land = 0.0
            for s in settled:
                if (px < s["x"]+s["l"]-1e-9 and px+pl > s["x"]+1e-9 and
                    py < s["y"]+s["w"]-1e-9 and py+pw > s["y"]+1e-9):
                    z_top = s["z"] + s["h"]
                    if z_top > z_land:
                        z_land = z_top
            if z_land + ph > CH + 1e-9:
                continue
            if z_land < 1e-9:
                p["z"] = 0.0
                settled.append(p)
                continue
            support_area = 0.0
            for s in settled:
                if abs((s["z"]+s["h"]) - z_land) < 1e-6:
                    ox = max(0.0, min(px+pl, s["x"]+s["l"]) - max(px, s["x"]))
                    oy = max(0.0, min(py+pw, s["y"]+s["w"]) - max(py, s["y"]))
                    support_area += ox * oy
            support_pct = support_area / base_area if base_area > 0 else 1.0
            if support_pct >= MIN_SUPPORT:
                p["z"] = z_land
                settled.append(p)
            else:
                # Try shifting to find better support
                supporters = [s for s in settled
                              if abs((s["z"]+s["h"])-z_land) < 1e-6
                              and px < s["x"]+s["l"]+pl and px+pl > s["x"]-pl
                              and py < s["y"]+s["w"]+pw and py+pw > s["y"]-pw]
                best_pos, best_sup = None, support_pct
                for s in supporters:
                    for tx in [s["x"], s["x"]+s["l"]-pl, px]:
                        for ty in [s["y"], s["y"]+s["w"]-pw, py]:
                            if tx < -1e-9 or ty < -1e-9: continue
                            if tx+pl > CL+1e-9 or ty+pw > CW+1e-9: continue
                            if any(tx < q["x"]+q["l"]-1e-9 and tx+pl > q["x"]+1e-9 and
                                   ty < q["y"]+q["w"]-1e-9 and ty+pw > q["y"]+1e-9 and
                                   z_land < q["z"]+q["h"]-1e-9 and z_land+ph > q["z"]+1e-9
                                   for q in settled):
                                continue
                            sa = sum(max(0.0, min(tx+pl, q["x"]+q["l"])-max(tx, q["x"])) *
                                     max(0.0, min(ty+pw, q["y"]+q["w"])-max(ty, q["y"]))
                                     for q in settled if abs((q["z"]+q["h"])-z_land) < 1e-6)
                            sp = sa / base_area if base_area > 0 else 1.0
                            if sp > best_sup:
                                best_sup = sp
                                best_pos = (tx, ty)
                if best_pos and best_sup >= MIN_SUPPORT:
                    p["x"], p["y"] = best_pos
                    p["z"] = z_land
                    settled.append(p)
                else:
                    if not any(px < s["x"]+s["l"]-1e-9 and px+pl > s["x"]+1e-9 and
                               py < s["y"]+s["w"]-1e-9 and py+pw > s["y"]+1e-9 and
                               0 < s["z"]+s["h"]-1e-9 and ph > s["z"]+1e-9
                               for s in settled):
                        if ph <= CH + 1e-9:
                            p["z"] = 0.0
                            settled.append(p)
        return settled

    # ──────────────────────────────────────────────────────────────────────
    # Decoder – shelf placer with auto-rotation + optional gravity
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def decode(chromosome, boxes, container, settle=False):
        """
        Shelf-based decoder. Chromosome is a permutation of box indices.
        For each box, ALL valid rotations are tried to find the best fit.
        Gravity settle is optional (skipped during SA loop for speed).
        """
        CL, CW, CH = container.L, container.W, container.H
        placed = []
        unplaced = []
        layers = []   # [{z, h, rows: [{y, w, x_next}]}]
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

        if settle:
            placed = SimulatedAnnealing._gravity_settle(placed, CL, CW, CH)

        return placed, unplaced

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
    # Main SA loop
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def solve(boxes, container, max_iterations=100000,
              initial_temp=1000.0, cooling_rate=0.99985,
              reheat_threshold=5000, reheat_ratio=0.5):
        """
        Simulated Annealing main loop.

        Parameters:
            max_iterations     – upper bound on iterations
            initial_temp       – starting temperature
            cooling_rate       – geometric cooling factor per iteration
            reheat_threshold   – iterations without improvement before reheat
            reheat_ratio       – fraction of initial_temp to reheat to

        Returns dict: {positions, utilization, fitness, iteration}
        """
        SA = SimulatedAnnealing

        # ── Initialise from best heuristic seed ───────────────────────
        current = SA.create_initial_chromosome(boxes, container)
        cur_placed, cur_unplaced = SA.decode(current, boxes, container)
        cur_fit = SA.compute_fitness(cur_placed, cur_unplaced, container)

        best      = list(current)
        best_fit  = cur_fit
        best_iter = 0
        temperature = initial_temp
        stagnation  = 0

        # ── Main loop ─────────────────────────────────────────────────
        for it in range(1, max_iterations + 1):
            neighbour = SA.perturb(current)
            n_placed, n_unplaced = SA.decode(neighbour, boxes, container)
            n_fit = SA.compute_fitness(n_placed, n_unplaced, container)

            if SA.should_accept(cur_fit, n_fit, temperature):
                current = neighbour
                cur_fit = n_fit

            if cur_fit > best_fit:
                best      = list(current)
                best_fit  = cur_fit
                best_iter = it
                stagnation = 0
            else:
                stagnation += 1

            if stagnation >= reheat_threshold:
                temperature = initial_temp * reheat_ratio
                stagnation = 0

            temperature *= cooling_rate
            if temperature < 1e-8:
                break

        # ── Final decode with gravity settle ──────────────────────────
        final_placed, _ = SA.decode(best, boxes, container, settle=True)
        vol = sum(p["l"]*p["w"]*p["h"] for p in final_placed)
        cvol = container.L * container.W * container.H
        util = (vol / cvol * 100) if cvol > 0 else 0.0

        return {
            "positions":   final_placed,
            "utilization": util,
            "fitness":     best_fit,
            "iteration":   best_iter,
        }


# ─────────────────────────── backward-compat shim ─────────────────
def solve(boxes, container):
    return SimulatedAnnealing.solve(boxes, container)
