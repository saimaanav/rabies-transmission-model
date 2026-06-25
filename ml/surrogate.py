"""
ml/surrogate.py — Fast ML Surrogate for the Agent-Based Model
=============================================================

The agent-based ``RabiesModel`` is the ground truth, but far too slow to
query thousands of times during policy search.  ``PolicySurrogate`` learns a
cheap approximation of it from a sample of runs (see ``ml/data.py``) so that
the optimizer can evaluate candidate policies in microseconds.

Model choice:
    Gradient-boosted regression trees (scikit-learn
    ``GradientBoostingRegressor``) are a strong default here — the
    policy-to-outcome surface is smooth but non-linear with interactions
    (e.g. vaccination and PEP are partial substitutes), which tree ensembles
    capture without feature engineering, and the datasets are small enough
    (hundreds to thousands of rows) that boosting trains in seconds.

One regressor is trained per outcome (``human_deaths``, ``attack_rate``).

Classes:
    PolicySurrogate -- fit / predict / cross-validate / persist the emulator.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

from ml.data import LEVERS, OUTCOMES


class PolicySurrogate:
    """
    A multi-output surrogate that predicts simulation outcomes from policy levers.

    Attributes:
        feature_names (list[str]): Lever columns used as model inputs.
        outcomes (list[str]): Outcome columns the surrogate predicts.
        models (dict): Mapping ``outcome -> fitted GradientBoostingRegressor``.
        cv_scores (dict): Mapping ``outcome -> mean R^2`` from cross-validation
            (populated by ``fit`` when ``cv`` is enabled).
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        outcomes: list[str] | None = None,
        n_estimators: int = 300,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 0,
    ):
        self.feature_names = (
            list(LEVERS) if feature_names is None else list(feature_names)
        )
        self.outcomes = list(OUTCOMES) if outcomes is None else list(outcomes)
        self._hparams = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
        )
        self.models: dict[str, GradientBoostingRegressor] = {}
        self.cv_scores: dict[str, float] = {}

    # -- Training -----------------------------------------------------------
    def fit(self, data: pd.DataFrame, cv: int | None = 5) -> "PolicySurrogate":
        """
        Train one regressor per outcome on the labelled dataset.

        Args:
            data (pandas.DataFrame): Must contain ``feature_names`` and the
                outcome columns.
            cv (int or None): If set, run ``cv``-fold cross-validation and store
                mean R^2 per outcome in ``cv_scores`` (skipped automatically if
                the dataset is too small to split).

        Returns:
            PolicySurrogate: ``self`` (fitted), for chaining.
        """
        X = data[self.feature_names].to_numpy(dtype=float)

        for outcome in self.outcomes:
            if outcome not in data.columns:
                continue
            y = data[outcome].to_numpy(dtype=float)
            model = GradientBoostingRegressor(**self._hparams)

            if cv and len(data) >= cv * 2:
                scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
                self.cv_scores[outcome] = float(np.mean(scores))

            model.fit(X, y)
            self.models[outcome] = model

        return self

    # -- Inference ----------------------------------------------------------
    def predict(self, policies: pd.DataFrame | dict) -> pd.DataFrame:
        """
        Predict outcomes for one or many policies.

        Args:
            policies (DataFrame or dict): Either a DataFrame with the lever
                columns, or a single ``{lever: value}`` dict.

        Returns:
            pandas.DataFrame: One row per input policy, columns = predicted
                outcomes. Predictions are clipped at zero (counts/rates are
                non-negative).
        """
        if isinstance(policies, dict):
            policies = pd.DataFrame([policies])

        X = policies[self.feature_names].to_numpy(dtype=float)
        preds = {
            name: np.clip(model.predict(X), 0.0, None)
            for name, model in self.models.items()
        }
        return pd.DataFrame(preds, index=policies.index)

    def feature_importances(self) -> pd.DataFrame:
        """
        Return per-lever feature importances for each fitted outcome model.

        Returns:
            pandas.DataFrame: Indexed by lever, one column per outcome.
        """
        return pd.DataFrame(
            {name: model.feature_importances_ for name, model in self.models.items()},
            index=self.feature_names,
        )

    # -- Persistence --------------------------------------------------------
    def save(self, path: str) -> None:
        """Serialise the fitted surrogate (models + metadata) to ``path``."""
        joblib.dump(
            {
                "feature_names": self.feature_names,
                "outcomes": self.outcomes,
                "hparams": self._hparams,
                "models": self.models,
                "cv_scores": self.cv_scores,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "PolicySurrogate":
        """Load a surrogate previously written by :meth:`save`."""
        blob = joblib.load(path)
        obj = cls(feature_names=blob["feature_names"], outcomes=blob["outcomes"])
        obj._hparams = blob["hparams"]
        obj.models = blob["models"]
        obj.cv_scores = blob.get("cv_scores", {})
        return obj
