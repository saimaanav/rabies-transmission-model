"""
ml/data.py — Training-Data Generation for the Policy Surrogate
==============================================================

This module turns the expensive agent-based ``RabiesModel`` into a labelled
dataset that a fast surrogate regressor can learn from.

Each training example is a single simulation run:

    * **Features**  — the policy levers a decision-maker can actually pull
      (dog vaccination coverage, human PEP access, and a public-education
      lever that lowers per-bite transmission).
    * **Labels**    — the epidemiological outcomes we care about
      (``human_deaths`` and ``attack_rate``) at the end of the run.

Because the surrogate only needs to learn the *shape* of the
policy-to-outcome response surface, training runs are executed at a reduced
but epidemiologically representative population (``DEFAULT_SIM_PARAMS``).
This keeps data generation tractable (hundreds of runs in minutes) while the
learned relationships transfer to the levers of interest.  All non-lever
parameters are held fixed so the surrogate isolates the effect of policy.

Sampling uses a space-filling Latin-Hypercube design so that, for a given
budget of runs, the lever space is covered far more evenly than independent
uniform draws would achieve.

Functions:
    run_simulation        -- Execute one agent-based run and return outcomes.
    sample_policies       -- Latin-Hypercube sample of the lever space.
    generate_training_data -- Build a full labelled DataFrame (optionally saved).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from model import RabiesModel

# ---------------------------------------------------------------------------
# Policy levers: the controllable decision variables and their valid ranges.
# These are the *features* the surrogate learns from and the optimizer
# searches over.  Ranges mirror the epidemiological bounds in params.py.
# ---------------------------------------------------------------------------
LEVERS = {
    # Fraction of dogs vaccinated (mass dog-vaccination campaign).
    "dog_vaccination_rate": (0.0, 0.9),
    # Fraction of bite victims with access to post-exposure prophylaxis.
    "human_pep_access_rate": (0.0, 1.0),
    # Per-bite transmission probability; public education / bite-avoidance
    # campaigns push this down, so it is treated as a (partly) controllable
    # lever rather than a fixed constant.
    "bite_transmission_prob": (0.05, 0.5),
}

# ---------------------------------------------------------------------------
# Outcome columns produced by each run (the surrogate's regression targets).
# ---------------------------------------------------------------------------
OUTCOMES = ["human_deaths", "attack_rate"]

# ---------------------------------------------------------------------------
# Fixed simulation configuration used for *training* runs.
#
# A reduced population keeps each run fast while preserving the dog/human
# ratio and contact dynamics that drive the policy response surface.  Only
# the LEVERS above are varied across runs; everything here stays constant.
# ---------------------------------------------------------------------------
DEFAULT_SIM_PARAMS = {
    "num_dogs": 2000,
    "num_humans": 20000,
    "initial_infected_dogs": 20,
    "initial_infected_humans": 0,
    "pep_survival_prob": 0.985,
    "dog_roaming_radius": 1.95,
    "bite_contact_radius": 1.0,
    "grid_size": 100,
    "incubation_period_range": (60, 90),
    "infectious_period_range": (7, 14),
    "simulation_days": 365,
    "human_mobility_radius": 3.25,
}


def run_simulation(
    policy: dict, sim_params: dict | None = None, random_seed: int | None = None
) -> dict:
    """
    Run a single agent-based simulation for a given policy and return outcomes.

    Args:
        policy (dict): Values for the LEVERS (any subset; missing levers fall
            back to ``sim_params``/model defaults).
        sim_params (dict or None): Fixed (non-lever) simulation configuration.
            Defaults to ``DEFAULT_SIM_PARAMS``.
        random_seed (int or None): Seed for reproducibility of the run.

    Returns:
        dict: ``{"human_deaths": int, "attack_rate": float}`` measured at the
            end of the simulation horizon.
    """
    sim_params = dict(DEFAULT_SIM_PARAMS if sim_params is None else sim_params)
    params = {**sim_params, **policy}

    model = RabiesModel(random_seed=random_seed, **params)
    for _ in range(model.simulation_days):
        model.step()

    stats = model.get_daily_statistics()
    return {
        "human_deaths": int(stats["humans_dead"]),
        "attack_rate": float(stats["attack_rate"]),
    }


def sample_policies(
    n_samples: int, levers: dict | None = None, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """
    Draw a space-filling Latin-Hypercube sample of the lever space.

    Latin-Hypercube sampling stratifies each lever into ``n_samples`` equal
    bins and draws exactly one point per bin, giving much more uniform
    coverage than independent uniform draws for the same number of runs.

    Args:
        n_samples (int): Number of policy points to generate.
        levers (dict or None): Mapping ``name -> (low, high)``. Defaults to LEVERS.
        rng (np.random.Generator or None): Random generator for reproducibility.

    Returns:
        pandas.DataFrame: One row per policy, one column per lever.
    """
    levers = LEVERS if levers is None else levers
    rng = np.random.default_rng() if rng is None else rng

    columns = {}
    for name, (low, high) in levers.items():
        # One stratified draw per bin, then shuffle to decorrelate dimensions.
        bins = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        rng.shuffle(bins)
        columns[name] = low + bins * (high - low)

    return pd.DataFrame(columns)


def generate_training_data(
    n_samples: int = 200,
    levers: dict | None = None,
    sim_params: dict | None = None,
    random_seed: int | None = 42,
    save_path: str | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Generate a labelled dataset of (policy levers -> outcomes) by running the
    agent-based model across a Latin-Hypercube sample of the lever space.

    Args:
        n_samples (int): Number of simulation runs / dataset rows.
        levers (dict or None): Lever ranges to sample. Defaults to LEVERS.
        sim_params (dict or None): Fixed simulation config. Defaults to
            DEFAULT_SIM_PARAMS.
        random_seed (int or None): Base seed; each run is seeded deterministically
            from it so the whole dataset is reproducible.
        save_path (str or None): If given, write the dataset to this CSV path.
        progress (bool): Print a lightweight progress line every few runs.

    Returns:
        pandas.DataFrame: Columns = lever names + OUTCOMES, ``n_samples`` rows.
    """
    levers = LEVERS if levers is None else levers
    rng = np.random.default_rng(random_seed)
    policies = sample_policies(n_samples, levers, rng)

    records = []
    for i, row in policies.iterrows():
        policy = row.to_dict()
        run_seed = None if random_seed is None else int(random_seed + i)
        outcomes = run_simulation(policy, sim_params, random_seed=run_seed)
        records.append({**policy, **outcomes})

        if progress and (i + 1) % max(1, n_samples // 20) == 0:
            print(f"  [data] completed {i + 1}/{n_samples} runs", flush=True)

    data = pd.DataFrame(records)

    if save_path:
        data.to_csv(save_path, index=False)
        if progress:
            print(f"  [data] saved {len(data)} rows -> {save_path}", flush=True)

    return data
