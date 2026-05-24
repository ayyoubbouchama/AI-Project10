"""
Run all three algorithms and save results to the results/ folder.

Usage:
    python run_all.py              # Run all algorithms
    python run_all.py greedy       # Run only greedy
    python run_all.py ga           # Run only GA
    python run_all.py sa           # Run only SA
"""

import sys
import os
import json
import time

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from core.data_loader import load_data

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "project10_container_dataset.json")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

os.makedirs(RESULTS_DIR, exist_ok=True)

boxes, container = load_data(DATA_PATH)
print(f"Loaded {len(boxes)} boxes, container {container.L}x{container.W}x{container.H} cm")
print(f"Container volume: {container.volume():,} cm3\n")

# Which algorithms to run
requested = sys.argv[1:] if len(sys.argv) > 1 else ["greedy", "ga", "sa"]
metrics = {}


# ── Greedy ────────────────────────────────────────────────────────
if "greedy" in requested:
    from algorithms.greedy import GreedyBFD

    print("Running Greedy BFD...")
    start = time.perf_counter()
    result = GreedyBFD.solve(boxes, container)
    elapsed = time.perf_counter() - start

    out = {
        "algorithm": "greedy",
        "utilization": round(result["utilization"], 2),
        "boxes_placed": len(result["positions"]),
        "total_boxes": len(boxes),
        "execution_time": round(elapsed, 3),
        "positions": result["positions"],
    }
    path = os.path.join(RESULTS_DIR, "solution_greedy.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    metrics["greedy"] = {
        "utilization": out["utilization"],
        "boxes_placed": out["boxes_placed"],
        "time": out["execution_time"],
    }
    print(f"  Utilization : {out['utilization']}%")
    print(f"  Boxes placed: {out['boxes_placed']}/{len(boxes)}")
    print(f"  Time        : {elapsed:.3f}s")
    print(f"  Saved to    : {path}\n")


# ── Genetic Algorithm ─────────────────────────────────────────────
if "ga" in requested:
    from algorithms.genetic import GeneticAlgorithm

    print("Running Genetic Algorithm (pop=20, gen=40, time_limit=40s)...")
    start = time.perf_counter()
    result = GeneticAlgorithm.solve(boxes, container)
    elapsed = time.perf_counter() - start

    out = {
        "algorithm": "ga",
        "utilization": round(result["utilization"], 2),
        "fitness": round(result["fitness"], 2),
        "boxes_placed": len(result["positions"]),
        "total_boxes": len(boxes),
        "execution_time": round(elapsed, 3),
        "positions": result["positions"],
    }
    path = os.path.join(RESULTS_DIR, "solution_ga.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    metrics["ga"] = {
        "utilization": out["utilization"],
        "boxes_placed": out["boxes_placed"],
        "time": out["execution_time"],
    }
    print(f"  Utilization : {out['utilization']}%")
    print(f"  Boxes placed: {out['boxes_placed']}/{len(boxes)}")
    print(f"  Fitness     : {out['fitness']}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Saved to    : {path}\n")


# ── Simulated Annealing ──────────────────────────────────────────
if "sa" in requested:
    from algorithms.simulated_annealing import SimulatedAnnealing

    print("Running Simulated Annealing (time_limit=40s)...")
    start = time.perf_counter()
    result = SimulatedAnnealing.solve(boxes, container)
    elapsed = time.perf_counter() - start

    out = {
        "algorithm": "sa",
        "utilization": round(result["utilization"], 2),
        "fitness": round(result["fitness"], 2),
        "boxes_placed": len(result["positions"]),
        "total_boxes": len(boxes),
        "execution_time": round(elapsed, 3),
        "best_iteration": result["iteration"],
        "positions": result["positions"],
    }
    path = os.path.join(RESULTS_DIR, "solution_sa.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    metrics["sa"] = {
        "utilization": out["utilization"],
        "boxes_placed": out["boxes_placed"],
        "time": out["execution_time"],
    }
    print(f"  Utilization : {out['utilization']}%")
    print(f"  Boxes placed: {out['boxes_placed']}/{len(boxes)}")
    print(f"  Fitness     : {out['fitness']}")
    print(f"  Time        : {elapsed:.2f}s")
    print(f"  Saved to    : {path}\n")


# ── Metrics summary ──────────────────────────────────────────────
if metrics:
    metrics_path = os.path.join(RESULTS_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'Algorithm':<22} {'Util':>8} {'Placed':>10} {'Time':>10}")
    print("=" * 54)
    for name, m in metrics.items():
        print(f"{name:<22} {m['utilization']:>7.2f}% "
              f"{m['boxes_placed']:>5}/{len(boxes)} {m['time']:>9.3f}s")
