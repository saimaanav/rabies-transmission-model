# Rabies Transmission Simulator & Policy Dashboard

A spatially explicit, stochastic agent-based model (ABM) of rabies transmission between free-roaming dogs and humans in Indian cities, built on [Mesa](https://mesa.readthedocs.io/) 3 and calibrated for Delhi and Kerala. A gradient-boosted surrogate trained on the ABM's outputs turns the simulator into a fast, budget-constrained policy optimizer, and an interactive [Streamlit](https://streamlit.io/) dashboard exposes both.

Received over $19k in compensation for model accuracy and insights.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Design](#model-design)
- [ML Policy Optimizer](#ml-policy-optimizer)
- [Configuration & Parameters](#configuration--parameters)
- [Policy Scenarios](#policy-scenarios)
- [Technical Details](#technical-details)
- [Acknowledgements](#acknowledgements)

---

## Overview

Dogs and humans are individual Mesa agents in a continuous 2D space. Each simulated day, every agent runs a fixed phase sequence: move (dogs random-walk within a roaming radius, humans are home-biased), infectious agents attempt bites on susceptible agents within a contact radius, then update disease state. A bite transmits with a configurable per-bite probability, and agents progress through the **SEIRD** compartments (Susceptible → Exposed → Infectious → Recovered → Dead). Rabies is always fatal in dogs. In humans, recovery requires PEP access and a successful Bernoulli draw on the PEP survival probability; otherwise the infection is fatal.

The model is calibrated against reported human death counts, 22 in Kerala and 45+ in Delhi. Calibration uses a closed-form death estimator and a parameter-grid sweep (`parameter_tuning.py`), so candidate parameters are ranked without running a full simulation per point. Spatial scaling maps simulation units to real geography (Delhi 1,483 km², Kerala 38,863 km²).

Full-scale runs take seconds to minutes, which rules out exhaustive policy search. The `ml/` package fits a surrogate to ABM outputs sampled with Latin-Hypercube designs over three policy levers (dog vaccination coverage, human PEP access, education-driven reduction in bite transmission). It then searches that surrogate for the lowest-mortality policy under a cost budget.

### Key Features

- **Agent-Based Modeling**: Individual-level stochastic simulation using the Mesa framework.
- **Spatial Dynamics**: Continuous 2D space with configurable roaming radii and contact distances.
- **SEIRD Compartmental Model**: Full disease progression with incubation and infectious periods.
- **Scalable Architecture**: Statistical sampling for populations exceeding 50,000 agents.
- **Interactive Dashboard**: Real-time parameter tuning, simulation execution, and result visualization via Streamlit.
- **Policy Analysis**: Pre-configured intervention scenarios with automated summary report generation.
- **Reproducibility**: Optional random seed control for deterministic results.
- **Calibration**: Analytical death estimator plus parameter-grid sweep to match reported Kerala and Delhi human death counts.
- **Surrogate Policy Optimization**: Gradient-boosted regressors (300 trees, 5-fold cross-validated R²) trained on Latin-Hypercube ABM runs, searched over 20,000 candidate policies per budget to produce a cost/deaths Pareto frontier.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                    │
│                      (app.py)                            │
│  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │
│  │ Sidebar  │  │ Sim Loop │  │  Visualization Tabs    │ │
│  │ Controls │→ │ & State  │→ │  (SEIRD, Epi, Spatial) │ │
│  └──────────┘  └──────────┘  └────────────────────────┘ │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │     RabiesModel         │
          │      (model.py)         │
          │  ┌───────┐ ┌─────────┐  │
          │  │ Dogs  │ │ Humans  │  │
          │  │ Agent │ │  Agent  │  │
          │  │ Pool  │ │  Pool   │  │
          │  └───┬───┘ └────┬────┘  │
          │  ┌───▼──────────▼────┐  │
          │  │  ContinuousSpace  │  │
          │  │  (Mesa Framework) │  │
          │  └───────────────────┘  │
          └─────────────────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
┌────▼─────┐   ┌──────▼──────┐   ┌──────▼──────┐
│ params.py│   │parameter_   │   │  utils.py   │
│ Defaults │   │tuning.py    │   │ Viz & Report│
│ & Ranges │   │ Calibration │   │  Generation │
└──────────┘   └─────────────┘   └─────────────┘
```

---

## Installation

### Prerequisites

- Python >= 3.11
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd RabiesTransmissionModel

# Install dependencies with uv (recommended)
uv sync

# Or install with pip
pip install mesa networkx numpy pandas streamlit plotly matplotlib kaleido scikit-learn joblib

# Launch the dashboard
streamlit run app.py
```

### Dependencies

| Package      | Version  | Purpose                              |
|------------- |----------|--------------------------------------|
| mesa         | >= 3.2.0 | Agent-based modeling framework       |
| numpy        | >= 2.3.1 | Numerical computation                |
| pandas       | >= 2.3.0 | Data manipulation and analysis       |
| streamlit    | >= 1.46.1| Interactive web dashboard            |
| plotly       | >= 6.2.0 | Interactive charting                 |
| matplotlib   | >= 3.10.3| Supplementary plotting               |
| kaleido      | >= 1.0.0 | Static image export for Plotly       |
| scikit-learn | >= 1.5.0 | ML surrogate model (policy optimizer)|
| joblib       | >= 1.4.0 | Surrogate model persistence          |
| networkx     | >= 3.3   | Required by Mesa's discrete space    |

---

## Usage

### Running the Dashboard

```bash
streamlit run app.py
```

1. **Configure Parameters** — Use the sidebar to set population sizes, vaccination rates, transmission probabilities, and spatial parameters.
2. **Load Presets** — Click "Load Delhi Urban Configuration" for pre-calibrated urban parameters.
3. **Run Simulation** — Click "Run Simulation" and monitor the progress bar.
4. **Analyze Results** — Explore five result tabs: SEIRD curves, epidemic curves, spatial heatmaps, raw data, and policy reports.
5. **Export Data** — Download simulation CSV data or policy summary reports.

### Programmatic Usage

```python
from model import RabiesModel
from params import DEFAULT_PARAMS

# Initialize the model with default parameters
model = RabiesModel(**DEFAULT_PARAMS)

# Run for 365 days
for day in range(365):
    model.step()
    stats = model.get_daily_statistics()
    print(f"Day {day+1}: {stats['humans_dead']} human deaths")
```

---

## Project Structure

```
RabiesTransmissionModel/
├── README.md                         # Project documentation
├── app.py                            # Streamlit dashboard (entry point)
├── pages/
│   └── 1_Policy_Optimizer.py         # Streamlit page: ML policy optimizer
├── model.py                          # Core ABM: RabiesModel class
├── agents.py                         # Agent definitions: DogAgent, HumanAgent
├── params.py                         # Default parameters, ranges, policy scenarios
├── parameter_tuning.py               # Calibration utilities for target outcomes
├── utils.py                          # Visualization and report generation
├── ml/                               # ML policy-optimization layer
│   ├── data.py                       # ABM training-data generation (LHS sampling)
│   ├── surrogate.py                  # Gradient-boosted surrogate emulator
│   ├── optimizer.py                  # Budget-constrained policy search
│   └── cli.py                        # `rabies-optimize` command-line interface
├── tests/
│   └── test_ml.py                    # Tests for the ML layer
└── pyproject.toml / uv.lock          # Dependency management
```

---

## Model Design

### Disease Progression (SEIRD)

```
  Bite Event
      │
      ▼
┌───────────┐  Incubation   ┌──────────┐  Infectious   ┌────────────┐
│Susceptible│ ─────────────→ │ Exposed  │ ────────────→ │ Infectious │
└───────────┘  (60-90 days)  └──────────┘  (7-14 days)  └─────┬──────┘
                                                               │
                                                ┌──────────────┼──────────────┐
                                          (PEP success)  (PEP failure)  (No PEP)
                                                │              │              │
                                                ▼              ▼              ▼
                                          ┌──────────┐   ┌─────────┐   ┌─────────┐
                                          │ Recovered│   │  Dead   │   │  Dead   │
                                          └──────────┘   └─────────┘   └─────────┘
```

### Performance Optimization

For populations exceeding 50,000 total agents, the model employs:

1. **Statistical Sampling**: Scales the population down while preserving epidemiological ratios.
2. **Spatial Sampling**: Checks a random subset of 1,000 agents for neighbor detection.
3. **Adaptive Grid Sizing**: Reduces grid dimensions for large populations to maintain density.
4. **Efficient State Counters**: Tracks compartment counts incrementally rather than re-counting.

---

## ML Policy Optimizer

The agent-based model is accurate but expensive — a single full-scale run can
take seconds to minutes, which makes exhaustive policy search impractical. The
`ml/` package adds a machine-learning layer that makes policy optimization fast.

### How it works

```
┌──────────────────┐   many runs    ┌────────────────────┐   learns    ┌──────────────────┐
│  RabiesModel     │ ─────────────▶ │  Training dataset  │ ──────────▶ │  Surrogate model │
│  (ground truth)  │  LHS sampling  │  levers → outcomes │  GB trees   │  (microsecond ƒ) │
└──────────────────┘                └────────────────────┘             └────────┬─────────┘
                                                                                 │ predicts
                                          ┌──────────────────────────────────────▼─────────┐
                                          │  PolicyOptimizer: minimise predicted deaths      │
                                          │  subject to  cost(policy) ≤ budget               │
                                          │  → best policy + cost/deaths Pareto frontier     │
                                          └──────────────────────────────────────────────────┘
```

1. **Data generation** (`ml/data.py`) — runs the ABM across a space-filling
   Latin-Hypercube sample of the three policy **levers** (dog vaccination
   coverage, human PEP access, and public-education-driven bite-transmission
   reduction), recording final human deaths and attack rate.
2. **Surrogate** (`ml/surrogate.py`) — gradient-boosted regression trees
   (scikit-learn) learn the `levers → outcomes` response surface and predict
   unseen policies in microseconds. Cross-validated R² is reported on training.
3. **Optimizer** (`ml/optimizer.py`) — searches the lever space against the
   surrogate to find the policy minimising predicted human deaths subject to a
   normalised **cost budget**, and returns the cost/deaths **Pareto frontier**.

### Command-line interface

A `rabies-optimize` console script is installed with the package:

```bash
# 1. Generate agent-based training data
rabies-optimize generate --n 300 --out ml_training_data.csv

# 2. Train the surrogate (prints cross-validated R² + feature importances)
rabies-optimize train --data ml_training_data.csv --out surrogate.joblib

# 3. Find the best policy within 60% of the maximum intervention budget
rabies-optimize optimize --model surrogate.joblib --budget 0.6
```

`optimize` will generate data and train a fresh surrogate automatically if no
`--model` is supplied.

### Programmatic usage

```python
from ml import generate_training_data, PolicySurrogate, PolicyOptimizer

data = generate_training_data(n_samples=300)          # run the ABM
surrogate = PolicySurrogate().fit(data)               # train the emulator
optimizer = PolicyOptimizer(surrogate)                # set up the search
result = optimizer.optimize(budget=0.6)               # constrained optimization

print(result.best_policy)        # {'dog_vaccination_rate': ..., ...}
print(result.predicted_deaths)   # surrogate-predicted human deaths
print(result.pareto)             # cost/deaths Pareto frontier (DataFrame)
```

### Interactive dashboard

Launch `streamlit run app.py` and open the **ML Policy Optimizer** page from
the sidebar to build the surrogate, inspect its accuracy and feature
importances, and explore the cost/deaths Pareto frontier with a budget slider.

### Cost model & budget

Each lever carries a normalised marginal cost (`ml.optimizer.DEFAULT_COSTS`):
sustained mass dog vaccination is the most resource-intensive, broad PEP access
is moderate, and education campaigns are comparatively cheap. The `budget` is on
a `0–1` scale where `1.0` affords maxing out every lever. Supply your own
`costs` dict to `PolicyOptimizer` to use real monetary values.

> **Note:** training runs use a reduced (but representative) population for
> tractability, so predicted death counts are on that scale — use them to
> **compare policies**, not as absolute national forecasts.

### Tests

```bash
uv run pytest tests/test_ml.py
```

---

## Configuration & Parameters

See `params.py` for the full parameter dictionary with descriptions, ranges, and bounds.

---

## Policy Scenarios

| Scenario              | Dog Vaccination | PEP Access | Notes                        |
|-----------------------|-----------------|------------|------------------------------|
| Baseline              | 10%             | 40%        | Current low-resource setting |
| Improved Vaccination  | 70%             | 40%        | Mass dog vaccination campaign|
| Improved PEP          | 10%             | 90%        | Better healthcare access     |
| Combined Intervention | 70%             | 90%        | Both strategies combined     |
| One Health Approach   | 80%             | 95%        | Comprehensive + education    |

---

## Technical Details

### Spatial Scaling

- **Delhi**: 1,483 km² → 40×40 grid (1 unit = 1 km)
- **Kerala**: 38,863 km² → 50×50 grid (4 units = 1 km)

### Reproducibility

Enable deterministic results by setting `random_seed=42` in the model constructor or toggling "Use Random Seed" in the Streamlit sidebar.

---

## Acknowledgements

- **Model Created By**: Maanav Chittireddy
- **Supported By**: Non Trivial
- **Framework**: [Mesa](https://mesa.readthedocs.io/) — Agent-Based Modeling in Python
- **Dashboard**: [Streamlit](https://streamlit.io/) — Data Apps Framework

---

## License

Released under the [MIT License](LICENSE).
