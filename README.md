# AI-Project10 — 3D Container Loading Optimizer

A university project that solves the **3D bin-packing problem** using three optimization algorithms. Given 386 boxes of 12 types and an ISO 20ft shipping container, the system finds arrangements that maximize volume utilization while respecting physical constraints (no overlaps, no boundary violations, gravity support, fragile-box orientation locking).

All three algorithms use **functionally equivalent shelf-based decoders** and **the same fitness metrics**, enabling scientifically valid performance comparison.

## Results

| Algorithm              | Utilization | Boxes Placed | Time     | Constraints |
|------------------------|-------------|--------------|----------|-------------|
| Greedy BFD             | 76.49%      | 360 / 386    | 0.1s     | All pass    |
| Genetic Algorithm      | 77.46%      | 364 / 386    | ~79s     | All pass    |
| Simulated Annealing    | 77.17%      | 366 / 386    | ~301s    | All pass    |

Theoretical maximum utilization: **80.96%**

## Constraints Enforced

- **No overlaps** — boxes cannot interpenetrate
- **Boundary** — all boxes fit within the container (589 x 234 x 239 cm)
- **Gravity** — every box rests on the floor or on another box
- **Fragile orientation** — 122 fragile boxes keep their vertical axis locked

## Algorithms

### 1. Greedy Best-Fit Decreasing (BFD)
Tries 12 heuristic orderings (volume, height, base area, weight, etc.) through the shared shelf decoder and keeps the best result. Deterministic, very fast.

### 2. Genetic Algorithm (GA)
Population-based evolutionary search over box orderings and rotations.
- **Crossover:** OX1, OX2, PMX (with proper mapping-table traversal)
- **Mutation:** Swap, inversion, rotation, scramble (adaptive pressure)
- **Features:** 12 heuristic seeds, 2-opt local search, stagnation restarts, elitism
- **Parameters:** pop=50, gen=100, seed=42

### 3. Simulated Annealing (SA)
Single-solution neighbourhood search with probabilistic acceptance.
- **Perturbations:** Swap, reverse segment, insert
- **Features:** Multi-restart on stagnation, calibrated temperature (T0=10, alpha=0.99954)
- **Parameters:** time_limit=300s, seed=42

## Unified Architecture

All algorithms use **functionally equivalent shelf-based decoders** with gravity enforcement:

| Algorithm | Decoder Function | Gravity Enforcement |
|-----------|-----------------|---------------------|
| Greedy    | `space_manager.place_boxes()` | `_gravity_drop()` in space_manager |
| GA        | `GeneticAlgorithm._decode_chromosome()` | `_gravity_settle()` post-processing |
| SA        | `SimulatedAnnealing.decode()` | `_gravity_drop()` post-processing |

All three decoders follow the same shelf-based strategy (O(n), no overlaps by construction) and produce physically valid packings. They share:
- **Stability metric:** `core.utils.compute_stability()` — weight x height formula
- **Fitness formula:** `utilization + 0.20 * stability + placed_bonus - 35.0 * unplaced`
- **Reproducibility:** `random.seed(42)` in both GA and SA

## Project Structure

```
AI-Project10/
├── algorithms/                   # Optimization algorithms
│   ├── __init__.py
│   ├── greedy.py                 #   Greedy BFD (12 heuristic orderings)
│   ├── genetic.py                #   Genetic Algorithm (OX1/OX2/PMX, adaptive)
│   └── simulated_annealing.py    #   Simulated Annealing (multi-restart)
├── core/                         # Shared infrastructure
│   ├── __init__.py
│   ├── models.py                 #   Box and Container classes
│   ├── data_loader.py            #   JSON dataset loader
│   ├── space_manager.py          #   Shelf decoder + gravity drop
│   └── utils.py                  #   Utilization and stability metrics
├── api/                          # Flask REST API
│   └── main.py                   #   /api/solve, /api/compare, /api/dataset
├── visualization/                # 3D web frontend (Three.js)
│   ├── index.html
│   ├── script.js
│   └── style.css
├── data/                         # Input dataset (JSON)
│   └── project10_container_dataset.json
├── results/                      # Output solution files
│   ├── solution_greedy.json
│   ├── solution_ga.json
│   ├── solution_sa.json
│   └── metrics.json
├── notebooks/                    # Jupyter analysis notebook
│   └── main.ipynb
├── run_all.py                    # Run all algorithms end-to-end
├── verify_solution.py            # Validate solutions (overlap, boundary, gravity)
├── requirements.txt              # Python dependencies
└── README.md
```

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Run All Algorithms
```bash
python run_all.py               # Run all three (~8 min total)
python run_all.py greedy        # Greedy only (~0.1s)
python run_all.py ga            # GA only (~2.5 min)
python run_all.py sa            # SA only (~5 min)
```

Results are saved to `results/` as JSON files.

### Verify Solutions
```bash
python verify_solution.py       # Check all solution files
```

### Start Web Application
```bash
python api/main.py
```
Open http://localhost:5000 for the interactive 3D visualization.

### API Endpoints

| Method | Endpoint         | Description                                    |
|--------|-----------------|------------------------------------------------|
| GET    | `/api/dataset`  | Returns box/container dataset (with fragile flags) |
| POST   | `/api/solve`    | Run a single algorithm                         |
| POST   | `/api/compare`  | Run all algorithms and compare                 |

## Dataset

- **Container:** ISO 20ft — 589 x 234 x 239 cm (volume: 32,940,414 cm3)
- **Boxes:** 386 individual boxes from 12 types (19-60 units per type)
- **Fragile:** 122 of 386 boxes are fragile and orientation-locked