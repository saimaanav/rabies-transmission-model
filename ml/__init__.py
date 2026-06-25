"""
ml — Machine-Learning Policy Optimization for the Rabies Transmission Model
===========================================================================

This subpackage adds a machine-learning layer on top of the agent-based
``RabiesModel``.  The agent-based model is accurate but expensive: a single
full-scale run can take seconds to minutes, which makes exhaustive policy
search (over vaccination coverage, PEP access, and public-education levers)
impractical.

The ML layer solves this with a two-stage pipeline:

    1. **Surrogate model** (``surrogate.PolicySurrogate``)
       A fast regressor (gradient-boosted trees) is trained on a sample of
       agent-based simulation runs.  It learns the mapping

           policy levers  ->  expected human deaths (and attack rate)

       and can then predict outcomes for *unseen* policies in microseconds.

    2. **Policy optimizer** (``optimizer.PolicyOptimizer``)
       Given the trained surrogate plus a per-lever cost model and a budget
       constraint, it searches the policy space for the intervention package
       that minimises predicted human deaths subject to ``cost <= budget``.
       It also returns the cost/deaths Pareto frontier for decision-making.

Typical workflow::

    from ml import generate_training_data, PolicySurrogate, PolicyOptimizer

    data = generate_training_data(n_samples=200)
    surrogate = PolicySurrogate().fit(data)
    optimizer = PolicyOptimizer(surrogate)
    result = optimizer.optimize(budget=0.6)
    print(result.best_policy, result.predicted_deaths)

See ``ml/cli.py`` (``rabies-optimize``) for a command-line interface.
"""

from ml.data import LEVERS, DEFAULT_SIM_PARAMS, generate_training_data, run_simulation
from ml.surrogate import PolicySurrogate
from ml.optimizer import DEFAULT_COSTS, PolicyOptimizer, OptimizationResult

__all__ = [
    "LEVERS",
    "DEFAULT_SIM_PARAMS",
    "DEFAULT_COSTS",
    "generate_training_data",
    "run_simulation",
    "PolicySurrogate",
    "PolicyOptimizer",
    "OptimizationResult",
]
