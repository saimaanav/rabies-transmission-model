"""
pages/1_Policy_Optimizer.py — Streamlit page for the ML Policy Optimizer
========================================================================

This Streamlit *multipage* view sits alongside the main simulator dashboard
(``app.py``). It drives the ``ml`` package end-to-end:

    1. Generate a sample of agent-based runs across the policy lever space.
    2. Train the gradient-boosted surrogate and report its accuracy.
    3. Search for the budget-constrained policy that minimises predicted
       human deaths, and visualise the cost/deaths Pareto frontier.

Launch with ``streamlit run app.py`` — Streamlit auto-discovers this page
from the ``pages/`` directory and shows it in the sidebar navigation.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ml import (
    DEFAULT_SIM_PARAMS,
    PolicyOptimizer,
    PolicySurrogate,
    generate_training_data,
)
from ml.optimizer import DEFAULT_COSTS

st.set_page_config(page_title="Rabies Policy Optimizer", page_icon="🧠", layout="wide")

st.title("🧠 ML Policy Optimizer")
st.markdown(
    "Train a fast **machine-learning surrogate** of the agent-based model, then "
    "search for the intervention package — dog vaccination, PEP access, and "
    "public-education (bite-transmission reduction) — that **minimises predicted "
    "human deaths under a budget constraint**."
)


@st.cache_data(show_spinner=False)
def _build_dataset(
    n_samples: int, num_humans: int, sim_days: int, seed: int
) -> pd.DataFrame:
    """Generate (and cache) a labelled training dataset of ABM runs."""
    sim_params = {
        **DEFAULT_SIM_PARAMS,
        "num_dogs": max(200, num_humans // 10),
        "num_humans": num_humans,
        "initial_infected_dogs": max(5, num_humans // 1000),
        "simulation_days": sim_days,
    }
    return generate_training_data(
        n_samples=n_samples, sim_params=sim_params, random_seed=seed, progress=False
    )


@st.cache_resource(show_spinner=False)
def _train_surrogate(n_samples: int, num_humans: int, sim_days: int, seed: int):
    """Train (and cache) the surrogate for a given data configuration."""
    data = _build_dataset(n_samples, num_humans, sim_days, seed)
    surrogate = PolicySurrogate().fit(data, cv=5)
    return surrogate, data


# ---------------------------------------------------------------------------
# Sidebar: training configuration
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ Surrogate Training")
n_samples = st.sidebar.slider(
    "Simulation runs (dataset size)",
    30,
    600,
    150,
    10,
    help="More runs = more accurate surrogate, slower to build.",
)
num_humans = st.sidebar.select_slider(
    "Population scale (humans per run)",
    options=[1000, 2000, 5000, 10000, 20000],
    value=5000,
    help="Reduced population keeps training tractable; relationships transfer.",
)
sim_days = st.sidebar.slider("Simulation days per run", 60, 365, 180, 5)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

st.sidebar.caption(
    f"≈ {n_samples} agent-based runs will be executed and cached on first use."
)

if not st.sidebar.button(
    "🚀 Build / Refresh Surrogate", type="primary", use_container_width=True
):
    if "optimizer_trained" not in st.session_state:
        st.info(
            "Configure the surrogate in the sidebar, then click **Build / Refresh Surrogate**."
        )
        st.stop()
else:
    st.session_state["optimizer_trained"] = True

with st.spinner(f"Running {n_samples} simulations and training the surrogate..."):
    surrogate, data = _train_surrogate(n_samples, num_humans, sim_days, int(seed))

# ---------------------------------------------------------------------------
# Surrogate diagnostics
# ---------------------------------------------------------------------------
st.subheader("📈 Surrogate Quality")
c1, c2, c3 = st.columns(3)
c1.metric("Training runs", len(data))
c2.metric(
    "CV R² — human deaths",
    f"{surrogate.cv_scores.get('human_deaths', float('nan')):.3f}",
)
c3.metric(
    "CV R² — attack rate", f"{surrogate.cv_scores.get('attack_rate', float('nan')):.3f}"
)
if surrogate.cv_scores.get("human_deaths", 0) < 0.5:
    st.caption(
        "⚠️ Low R²: increase the number of simulation runs for a more reliable surrogate."
    )

imp = surrogate.feature_importances()["human_deaths"].sort_values()
st.plotly_chart(
    px.bar(
        imp,
        orientation="h",
        labels={"value": "Importance", "index": "Lever"},
        title="What drives human deaths? (surrogate feature importance)",
    ),
    use_container_width=True,
)

# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
st.subheader("🎯 Constrained Policy Optimization")
optimizer = PolicyOptimizer(surrogate)

cost_str = ", ".join(f"{k}={v}" for k, v in DEFAULT_COSTS.items())
budget = st.slider(
    "Budget (fraction of maximum intervention spend)",
    0.0,
    1.0,
    0.6,
    0.05,
    help=f"Normalised cost weights — {cost_str}",
)

result = optimizer.optimize(budget=budget)

st.markdown(f"#### Recommended policy under a **{budget:.0%}** budget")
cols = st.columns(len(result.best_policy) + 1)
labels = {
    "dog_vaccination_rate": "💉 Dog vaccination",
    "human_pep_access_rate": "🏥 PEP access",
    "bite_transmission_prob": "📢 Bite transmission",
}
for col, (lever, value) in zip(cols, result.best_policy.items()):
    col.metric(
        labels.get(lever, lever),
        f"{value:.0%}" if lever != "bite_transmission_prob" else f"{value:.2f}",
    )
cols[-1].metric("☠️ Predicted deaths", f"{result.predicted_deaths:.1f}")

# Pareto frontier: deaths vs cost.
frontier = result.pareto
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=result.evaluated["cost"],
        y=result.evaluated["predicted_deaths"],
        mode="markers",
        name="Candidate policies",
        marker=dict(size=4, color="lightgray", opacity=0.4),
    )
)
fig.add_trace(
    go.Scatter(
        x=frontier["cost"],
        y=frontier["predicted_deaths"],
        mode="lines+markers",
        name="Pareto frontier",
        line=dict(color="#667eea", width=3),
    )
)
fig.add_trace(
    go.Scatter(
        x=[result.cost],
        y=[result.predicted_deaths],
        mode="markers",
        name="Recommended (within budget)",
        marker=dict(size=14, color="#e74c3c", symbol="star"),
    )
)
fig.add_vline(
    x=budget * optimizer.max_cost,
    line_dash="dash",
    line_color="#2ecc71",
    annotation_text="Budget",
)
fig.update_layout(
    title="Cost vs Predicted Human Deaths",
    xaxis_title="Normalised intervention cost",
    yaxis_title="Predicted human deaths",
    height=480,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("##### Pareto-efficient policies (cheapest first)")
st.dataframe(
    frontier[list(result.best_policy) + ["cost", "predicted_deaths"]].round(3),
    use_container_width=True,
)
st.caption(
    "The surrogate is trained at a reduced population for speed, so predicted "
    "death counts are on that scale — use them to compare policies, not as "
    "absolute national forecasts."
)
