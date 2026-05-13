"""
Verify solution files in results/ for physical constraint violations.

Checks:
  1. Boundary  — every box fits within the container dimensions
  2. Overlap   — no two boxes interpenetrate
  3. Floating  — every box rests on the floor or on another box (gravity)

Usage:
    python verify_solution.py                   # verify all solution_*.json
    python verify_solution.py results/solution_greedy.json   # verify one file
"""

import sys
import os
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Container dimensions (ISO 20ft)
CONTAINER_L = 589
CONTAINER_W = 234
CONTAINER_H = 239

EPS = 1.0  # tolerance in cm


def check_boundary(positions, CL, CW, CH):
    """Check that every box fits within container bounds."""
    violations = []
    for p in positions:
        x, y, z = p["x"], p["y"], p["z"]
        l, w, h = p["l"], p["w"], p["h"]
        if (x < -EPS or y < -EPS or z < -EPS or
                x + l > CL + EPS or y + w > CW + EPS or z + h > CH + EPS):
            violations.append(p["id"])
    return violations


def check_overlap(positions):
    """Check that no two boxes overlap (O(n²) pairwise)."""
    violations = []
    n = len(positions)
    for i in range(n):
        a = positions[i]
        for j in range(i + 1, n):
            b = positions[j]
            # Two boxes overlap if they overlap on ALL three axes
            if not (a["x"] + a["l"] <= b["x"] + EPS or
                    b["x"] + b["l"] <= a["x"] + EPS or
                    a["y"] + a["w"] <= b["y"] + EPS or
                    b["y"] + b["w"] <= a["y"] + EPS or
                    a["z"] + a["h"] <= b["z"] + EPS or
                    b["z"] + b["h"] <= a["z"] + EPS):
                violations.append((a["id"], b["id"]))
    return violations


def check_floating(positions):
    """Check gravity: every box must rest on the floor or on another box."""
    violations = []
    for p in positions:
        if p["z"] < EPS:
            continue  # on the floor — OK

        supported = False
        for q in positions:
            if q is p:
                continue
            # q's top face touches p's bottom face
            if abs((q["z"] + q["h"]) - p["z"]) < EPS:
                # and they overlap in x-y
                if (p["x"] < q["x"] + q["l"] - 0.01 and
                        p["x"] + p["l"] > q["x"] + 0.01 and
                        p["y"] < q["y"] + q["w"] - 0.01 and
                        p["y"] + p["w"] > q["y"] + 0.01):
                    supported = True
                    break
        if not supported:
            violations.append(p["id"])
    return violations


def verify_file(filepath):
    """Run all checks on a single solution JSON file."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    algo = data.get("algorithm", os.path.basename(filepath))
    positions = data["positions"]
    total = data.get("total_boxes", len(positions))
    util = data.get("utilization", 0)

    print(f"\n{'=' * 60}")
    print(f"  {algo.upper()}  —  {os.path.basename(filepath)}")
    print(f"  Boxes placed: {len(positions)} / {total}")
    print(f"  Utilization:  {util:.2f}%")
    print(f"{'=' * 60}")

    # 1. Boundary
    boundary = check_boundary(positions, CONTAINER_L, CONTAINER_W, CONTAINER_H)
    if boundary:
        print(f"  [FAIL] BOUNDARY VIOLATIONS: {len(boundary)} boxes")
        print(f"    IDs: {boundary[:10]}{'...' if len(boundary) > 10 else ''}")
    else:
        print(f"  [PASS] Boundary:  OK  (all {len(positions)} boxes within container)")

    # 2. Overlap
    overlap = check_overlap(positions)
    if overlap:
        print(f"  [FAIL] OVERLAP VIOLATIONS: {len(overlap)} pairs")
        for a, b in overlap[:5]:
            print(f"    Box {a} overlaps Box {b}")
        if len(overlap) > 5:
            print(f"    ... and {len(overlap) - 5} more")
    else:
        print(f"  [PASS] Overlap:   OK  (0 intersecting pairs)")

    # 3. Floating (gravity)
    floating = check_floating(positions)
    if floating:
        print(f"  [FAIL] FLOATING VIOLATIONS: {len(floating)} boxes")
        print(f"    IDs: {floating[:10]}{'...' if len(floating) > 10 else ''}")
    else:
        print(f"  [PASS] Gravity:   OK  (all boxes supported)")

    # Summary
    passed = not boundary and not overlap and not floating
    status = "PASS" if passed else "FAIL"
    print(f"\n  Result: {status}")
    return passed


def main():
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        files = sorted(glob.glob(os.path.join(RESULTS_DIR, "solution_*.json")))

    if not files:
        print("No solution files found in results/")
        return

    print(f"Verifying {len(files)} solution file(s)...")

    results = {}
    for f in files:
        if not os.path.exists(f):
            print(f"  File not found: {f}")
            continue
        results[f] = verify_file(f)

    # Final summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    all_pass = True
    for f, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {os.path.basename(f):30s}  {status}")
        if not passed:
            all_pass = False

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")


if __name__ == "__main__":
    main()
