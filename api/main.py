"""
Flask API for the 3D Bin Packing Tool
======================================

Endpoints:
  POST /api/solve    — Run a single algorithm on the given boxes/container.
  POST /api/compare  — Run all available algorithms and return comparison.
  GET  /api/dataset  — Return the default synthetic dataset.
  GET  /              — Serve the visualization frontend.
"""

import os
import sys
import json
import time

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ── Make sure project root is on sys.path so imports work ────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from core.models import Box, Container
from core.space_manager import place_boxes
from core.utils import compute_utilization

# ── App setup ────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=os.path.join(PROJECT_ROOT, "visualization"),
    static_url_path="/static",
)
CORS(app)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")


# ──────────────────────── helpers ────────────────────────────────

def _parse_boxes(raw_boxes):
    """Convert list of dicts from JSON into Box objects."""
    boxes = []
    for b in raw_boxes:
        boxes.append(Box(
            id=b.get("id", len(boxes) + 1),
            l=float(b["length"]),
            w=float(b["width"]),
            h=float(b["height"]),
            weight=float(b.get("weight", 1)),
            fragile=b.get("fragile", False),
            orientation_locked=b.get("orientation_locked", False),
        ))
    return boxes


def _parse_container(raw):
    """Convert dict from JSON into a Container object."""
    return Container(
        L=float(raw["length"]),
        W=float(raw["width"]),
        H=float(raw["height"]),
    )


def _run_algorithm(name, boxes, container):
    """
    Run the named algorithm on the given boxes/container.
    Returns (result_dict, elapsed_seconds).
    """
    start = time.perf_counter()

    if name == "ga":
        from algorithms.genetic import solve as ga_solve
        result = ga_solve(boxes, container)
    elif name == "greedy":
        from algorithms.greedy import solve
        result = solve(boxes, container)
    elif name == "sa":
        from algorithms.simulated_annealing import SimulatedAnnealing
        # Use 40s time limit (matches optimized default)
        result = SimulatedAnnealing.solve(boxes, container, time_limit=40.0)
    else:
        raise ValueError(f"Unknown algorithm: {name}")

    elapsed = time.perf_counter() - start
    return result, elapsed


def _format_result(result, elapsed, algorithm_name):
    """Build a JSON-friendly result dict."""
    positions = result["positions"]
    return {
        "algorithm": algorithm_name,
        "utilization": round(result["utilization"], 2),
        "boxesPlaced": len(positions),
        "totalBoxes": result.get("totalBoxes", len(positions)),
        "executionTime": round(elapsed, 3),
        "positions": positions,
        "fitness": round(result.get("fitness", result["utilization"]), 2),
        "extra": {
            k: v for k, v in result.items()
            if k not in ("positions", "utilization", "fitness")
        },
    }


# ──────────────────────── routes ─────────────────────────────────

@app.route("/")
def index():
    """Serve the main visualization page."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/dataset", methods=["GET"])
def get_dataset():
    """Return the default dataset (boxes + container) as JSON."""
    # Read boxes.json
    json_path = os.path.join(DATA_DIR, "project10_container_dataset.json")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Expand box_types × quantity into individual box entries
    boxes = []
    box_id = 1
    for bt in data["box_types"]:
        qty = bt.get("quantity", 1)
        for _ in range(qty):
            boxes.append({
                "id": box_id,
                "length": float(bt["length_cm"]),
                "width": float(bt["width_cm"]),
                "height": float(bt["height_cm"]),
                "weight": float(bt["weight_kg"]),
                "fragile": bt.get("fragile", False),
                "orientation_locked": bt.get("orientation_locked", False),
            })
            box_id += 1

    # Read container from boxes.json
    c = data["container"]["inner_dimensions"]
    container = {
        "name": data["container"].get("type", "Container"),
        "length": c["length_cm"],
        "width": c["width_cm"],
        "height": c["height_cm"],
        "unit": "cm",
    }

    return jsonify({"boxes": boxes, "container": container})


@app.route("/api/solve", methods=["POST"])
def solve():
    """
    Run a single algorithm.

    Request body:
    {
      "boxes": [{ "id":1, "length":60, "width":40, "height":30, "weight":12 }, ...],
      "container": { "length":590, "width":235, "height":239 },
      "algorithm": "ga" | "greedy" | "sa"
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        boxes = _parse_boxes(data["boxes"])
        container = _parse_container(data["container"])
        algo = data.get("algorithm", "ga")

        result, elapsed = _run_algorithm(algo, boxes, container)
        result["totalBoxes"] = len(boxes)

        return jsonify(_format_result(result, elapsed, algo))
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/compare", methods=["POST"])
def compare():
    """
    Run all available algorithms on the same input and return comparison.

    Request body: same as /api/solve but without "algorithm" field.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    try:
        boxes_data = data["boxes"]
        container_data = data["container"]
        container = _parse_container(container_data)

        results = []
        algorithms = []

        # Detect which algorithms are available
        try:
            from algorithms.genetic import solve as _  # noqa
            algorithms.append("ga")
        except ImportError:
            pass
        try:
            from algorithms.greedy import solve as _  # noqa
            algorithms.append("greedy")
        except ImportError:
            pass
        try:
            from algorithms.simulated_annealing import solve as _  # noqa
            algorithms.append("sa")
        except ImportError:
            pass

        if not algorithms:
            return jsonify({"error": "No algorithms available"}), 500

        for algo in algorithms:
            boxes = _parse_boxes(boxes_data)  # fresh copy each time
            result, elapsed = _run_algorithm(algo, boxes, container)
            result["totalBoxes"] = len(boxes)
            results.append(_format_result(result, elapsed, algo))

        return jsonify({"results": results})
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ──────────────────────── main ───────────────────────────────────

if __name__ == "__main__":
    print("\n  >>  3D Bin Packing API running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
