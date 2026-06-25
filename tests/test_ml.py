"""
Tests for the ML policy-optimization layer (``ml`` package).

These use a deliberately small/fast simulation configuration so the whole
suite runs in a few seconds while still exercising the real agent-based model,
the surrogate training path, persistence, and the constrained optimizer.
"""

import pandas as pd
import pytest

from ml import (
    DEFAULT_SIM_PARAMS,
    LEVERS,
    PolicyOptimizer,
    PolicySurrogate,
    generate_training_data,
    run_simulation,
)

# Tiny config: keeps each agent-based run to a fraction of a second.
FAST_SIM = {
    **DEFAULT_SIM_PARAMS,
    "num_dogs": 200,
    "num_humans": 1000,
    "initial_infected_dogs": 8,
    "simulation_days": 60,
}


@pytest.fixture(scope="module")
def dataset():
    return generate_training_data(
        n_samples=20, sim_params=FAST_SIM, random_seed=7, progress=False
    )


def test_run_simulation_outcomes():
    out = run_simulation(
        {
            "dog_vaccination_rate": 0.0,
            "human_pep_access_rate": 0.0,
            "bite_transmission_prob": 0.4,
        },
        sim_params=FAST_SIM,
        random_seed=3,
    )
    assert set(out) == {"human_deaths", "attack_rate"}
    assert out["human_deaths"] >= 0
    assert 0.0 <= out["attack_rate"] <= 1.0


def test_run_simulation_is_reproducible():
    policy = {
        "dog_vaccination_rate": 0.2,
        "human_pep_access_rate": 0.5,
        "bite_transmission_prob": 0.3,
    }
    a = run_simulation(policy, sim_params=FAST_SIM, random_seed=11)
    b = run_simulation(policy, sim_params=FAST_SIM, random_seed=11)
    assert a == b


def test_generate_training_data_schema(dataset):
    assert len(dataset) == 20
    for lever in LEVERS:
        assert lever in dataset.columns
    assert {"human_deaths", "attack_rate"}.issubset(dataset.columns)


def test_sampled_policies_respect_bounds(dataset):
    for lever, (low, high) in LEVERS.items():
        assert dataset[lever].min() >= low - 1e-9
        assert dataset[lever].max() <= high + 1e-9


def test_surrogate_fit_and_predict(dataset):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    preds = surrogate.predict(dataset)
    assert list(preds.columns) == ["human_deaths", "attack_rate"]
    assert (preds["human_deaths"] >= 0).all()  # clipped non-negative
    assert len(preds) == len(dataset)


def test_surrogate_single_policy_dict(dataset):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    preds = surrogate.predict(
        {
            "dog_vaccination_rate": 0.5,
            "human_pep_access_rate": 0.5,
            "bite_transmission_prob": 0.2,
        }
    )
    assert len(preds) == 1


def test_surrogate_save_load_roundtrip(dataset, tmp_path):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    path = tmp_path / "surrogate.joblib"
    surrogate.save(str(path))
    loaded = PolicySurrogate.load(str(path))
    pd.testing.assert_frame_equal(surrogate.predict(dataset), loaded.predict(dataset))


def test_optimizer_respects_budget(dataset):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    optimizer = PolicyOptimizer(surrogate)
    result = optimizer.optimize(budget=0.4, n_candidates=3000, random_state=0)
    # Recommended policy must be within budget and within lever bounds.
    assert result.cost <= 0.4 * optimizer.max_cost + 1e-6
    for lever, (low, high) in LEVERS.items():
        assert low - 1e-9 <= result.best_policy[lever] <= high + 1e-9
    assert result.predicted_deaths >= 0


def test_optimizer_pareto_is_monotone(dataset):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    optimizer = PolicyOptimizer(surrogate)
    result = optimizer.optimize(budget=1.0, n_candidates=3000, random_state=0)
    frontier = result.pareto
    # Along the frontier, cost increases and predicted deaths strictly decrease.
    assert frontier["cost"].is_monotonic_increasing
    assert frontier["predicted_deaths"].is_monotonic_decreasing


def test_optimizer_infeasible_budget_raises(dataset):
    surrogate = PolicySurrogate().fit(dataset, cv=None)
    optimizer = PolicyOptimizer(surrogate)
    with pytest.raises(ValueError):
        optimizer.optimize(budget=-0.1)
