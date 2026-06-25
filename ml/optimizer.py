"""
ml/optimizer.py — ML-Driven Policy Optimization
===============================================

Given a trained :class:`~ml.surrogate.PolicySurrogate`, this module searches
the policy space for the intervention package that minimises predicted human
deaths subject to a budget constraint.

Problem statement::

    minimise    predicted_human_deaths(policy)        (via the surrogate)
    over         policy in LEVERS ranges
    subject to   cost(policy) <= budget

Cost model:
    Each lever carries a normalised marginal cost in ``DEFAULT_COSTS`` (cost
    of moving the lever across its *full* range). ``budget`` is expressed on
    the same 0-1 scale, where 1.0 is the cost of maxing out every lever. This
    keeps the interface simple and unit-free; callers can substitute real
    monetary costs by supplying their own ``costs`` dict.

The search is a surrogate-evaluated random search (cheap because the
surrogate is microsecond-fast), which also yields the cost/deaths **Pareto
frontier** — the set of policies for which no cheaper policy achieves fewer
predicted deaths — for transparent decision-making.

Classes:
    OptimizationResult -- structured result (best policy, frontier, etc.).
    PolicyOptimizer    -- runs the constrained search over the surrogate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ml.data import LEVERS
from ml.surrogate import PolicySurrogate

# ---------------------------------------------------------------------------
# Normalised marginal cost of driving each lever across its full range.
#
# These defaults encode rough real-world intuition: sustained mass dog
# vaccination is the most resource-intensive lever, broad PEP access is
# moderately expensive, and public-education campaigns that reduce bite
# transmission are comparatively cheap. Override with real costs as needed.
# ---------------------------------------------------------------------------
DEFAULT_COSTS = {
    "dog_vaccination_rate": 1.0,
    "human_pep_access_rate": 0.7,
    "bite_transmission_prob": 0.4,  # cost of pushing transmission to its floor
}


@dataclass
class OptimizationResult:
    """
    Result of a constrained policy optimization.

    Attributes:
        best_policy (dict): Lever values of the recommended policy.
        predicted_deaths (float): Surrogate-predicted human deaths for it.
        cost (float): Normalised cost of the recommended policy.
        budget (float): Budget constraint that was applied.
        pareto (pandas.DataFrame): Cost/deaths Pareto-efficient policies,
            sorted by ascending cost.
        evaluated (pandas.DataFrame): All sampled candidate policies with their
            predicted deaths and cost (useful for plotting / inspection).
        target (str): Outcome that was minimised.
    """

    best_policy: dict
    predicted_deaths: float
    cost: float
    budget: float
    pareto: pd.DataFrame = field(repr=False)
    evaluated: pd.DataFrame = field(repr=False)
    target: str = "human_deaths"


class PolicyOptimizer:
    """
    Optimise intervention policies against a trained surrogate.

    Attributes:
        surrogate (PolicySurrogate): Fitted outcome emulator.
        levers (dict): Lever ranges to search over.
        costs (dict): Per-lever normalised costs.
        target (str): Outcome to minimise (default ``"human_deaths"``).
    """

    def __init__(
        self,
        surrogate: PolicySurrogate,
        levers: dict | None = None,
        costs: dict | None = None,
        target: str = "human_deaths",
    ):
        self.surrogate = surrogate
        self.levers = LEVERS if levers is None else levers
        self.costs = DEFAULT_COSTS if costs is None else costs
        self.target = target

    # -- Cost model ---------------------------------------------------------
    def policy_cost(self, policies: pd.DataFrame) -> np.ndarray:
        """
        Normalised cost of each policy in [0, len(levers)] before scaling.

        For each lever, cost is the fraction of its range that is "spent".
        For ``bite_transmission_prob`` the cost is inverted: a *lower*
        transmission probability is the expensive outcome (education effort),
        so spending equals how far transmission is pushed below its ceiling.

        Returns:
            numpy.ndarray: Total normalised cost per policy.
        """
        total = np.zeros(len(policies), dtype=float)
        for name, (low, high) in self.levers.items():
            if name not in policies.columns or name not in self.costs:
                continue
            frac = (policies[name].to_numpy(dtype=float) - low) / (high - low)
            if name == "bite_transmission_prob":
                frac = 1.0 - frac  # lower transmission = more effort spent
            total += self.costs[name] * frac
        return total

    @property
    def max_cost(self) -> float:
        """Cost of maxing out every priced lever (used to scale the budget)."""
        return float(sum(self.costs.get(name, 0.0) for name in self.levers))

    # -- Search -------------------------------------------------------------
    def optimize(
        self,
        budget: float = 1.0,
        n_candidates: int = 20000,
        random_state: int | None = 0,
    ) -> OptimizationResult:
        """
        Find the budget-feasible policy with the fewest predicted deaths.

        Args:
            budget (float): Cost ceiling on the same 0-1 scale as the cost
                model (1.0 = afford everything; 0.0 = do nothing).
            n_candidates (int): Number of random candidate policies to evaluate
                through the surrogate.
            random_state (int or None): Seed for the candidate sampler.

        Returns:
            OptimizationResult: Recommended policy plus the Pareto frontier.
        """
        rng = np.random.default_rng(random_state)

        # Sample candidate policies uniformly across the lever ranges.
        cols = {
            name: rng.uniform(low, high, n_candidates)
            for name, (low, high) in self.levers.items()
        }
        candidates = pd.DataFrame(cols)

        # Evaluate predicted target + cost for every candidate (vectorised).
        predicted = self.surrogate.predict(candidates)[self.target].to_numpy()
        cost = self.policy_cost(candidates)
        budget_abs = budget * self.max_cost

        evaluated = candidates.copy()
        evaluated["predicted_deaths"] = predicted
        evaluated["cost"] = cost

        feasible = evaluated[evaluated["cost"] <= budget_abs + 1e-9]
        if feasible.empty:
            raise ValueError(
                f"No feasible policy under budget={budget}. "
                f"Try a larger budget (max useful cost = {self.max_cost:.2f})."
            )

        best = feasible.loc[feasible["predicted_deaths"].idxmin()]
        best_policy = {name: float(best[name]) for name in self.levers}

        return OptimizationResult(
            best_policy=best_policy,
            predicted_deaths=float(best["predicted_deaths"]),
            cost=float(best["cost"]),
            budget=budget,
            pareto=self._pareto_frontier(evaluated),
            evaluated=evaluated,
            target=self.target,
        )

    @staticmethod
    def _pareto_frontier(evaluated: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the cost/deaths Pareto frontier from evaluated candidates.

        A policy is Pareto-efficient if no other policy is both cheaper *and*
        achieves fewer (or equal) predicted deaths.

        Returns:
            pandas.DataFrame: Frontier policies sorted by ascending cost.
        """
        ordered = evaluated.sort_values(["cost", "predicted_deaths"]).reset_index(
            drop=True
        )
        frontier = []
        best_deaths = np.inf
        for _, row in ordered.iterrows():
            if row["predicted_deaths"] < best_deaths - 1e-9:
                frontier.append(row)
                best_deaths = row["predicted_deaths"]
        return pd.DataFrame(frontier).reset_index(drop=True)
