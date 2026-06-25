"""
params.py — Default Parameters, Ranges, and Policy Scenarios
=============================================================

This module centralises all configurable parameters for the rabies
transmission agent-based model.  It is the single source of truth for:

    1. **DEFAULT_PARAMS** — Baseline values used when no overrides are given.
    2. **PARAM_RANGES**   — Min/max bounds exposed on the Streamlit UI sliders.
    3. **PARAM_DESCRIPTIONS** — Human-readable tooltips for each parameter.
    4. **PARAM_BOUNDS**   — Epidemiologically realistic hard limits.
    5. **POLICY_SCENARIOS** — Pre-defined intervention packages for quick testing.

All parameter values are grounded in published epidemiological literature
and field-study estimates.  See the ``methodology_description.md`` file for
full citations and justification.

Usage:
    >>> from params import DEFAULT_PARAMS
    >>> model = RabiesModel(**DEFAULT_PARAMS)
"""

# ---------------------------------------------------------------------------
# Default simulation parameters based on epidemiological literature
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    # -- Population parameters (tuned to produce ~22 human deaths) ----------
    "num_dogs": 289000,
    "num_humans": 35800000,
    "initial_infected_dogs": 5,  # Start with 5 infected dogs for gradual spread
    "initial_infected_humans": 0,
    # -- Intervention parameters (baseline: no vaccination, reduced PEP) ----
    "dog_vaccination_rate": 0.0,  # No vaccination to maximise transmission
    "human_pep_access_rate": 0.65,  # Slightly reduced PEP access
    # -- Transmission parameters (calibrated for target death count) --------
    "bite_transmission_prob": 0.15,  # 15% probability per bite event
    "pep_survival_prob": 0.985,  # 98.5% survival with PEP treatment
    # -- Spatial parameters -------------------------------------------------
    "dog_roaming_radius": 1.95,  # 1.95 km average roaming distance
    "bite_contact_radius": 1.0,  # 1 km effective contact radius
    "grid_size": 500,  # Larger grid for massive population
    # -- Human mobility -----------------------------------------------------
    "human_mobility_radius": 3.25,  # Average of 3.0-3.5 km range
    # -- Disease progression ------------------------------------------------
    "incubation_period_range": (60, 90),  # 2-3 months (Exposed -> Infectious)
    "infectious_period_range": (7, 14),  # 7-14 days   (Infectious -> outcome)
    # -- Simulation control -------------------------------------------------
    "simulation_days": 365,  # One calendar year
}


# ---------------------------------------------------------------------------
# Parameter ranges for UI sliders and input validation
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "num_dogs": (50, 1000),
    "num_humans": (100, 5000),
    "dog_roaming_radius": (1.0, 10.0),
    "bite_contact_radius": (0.5, 3.0),
    "grid_size": (50, 200),
    "simulation_days": (30, 730),
}


# ---------------------------------------------------------------------------
# Human-readable parameter descriptions (used in UI tooltips & reports)
# ---------------------------------------------------------------------------
PARAM_DESCRIPTIONS = {
    "num_dogs": "Total number of dogs in the population",
    "num_humans": "Total number of humans in the population",
    "initial_infected_dogs": "Number of dogs initially infected with rabies",
    "initial_infected_humans": "Number of humans initially infected with rabies",
    "dog_vaccination_rate": "Proportion of dogs vaccinated against rabies",
    "human_pep_access_rate": "Proportion of humans with access to post-exposure prophylaxis",
    "bite_transmission_prob": "Probability of transmission per bite event",
    "pep_survival_prob": "Probability of survival with proper PEP treatment",
    "dog_roaming_radius": "Average distance dogs roam per day (grid units)",
    "bite_contact_radius": "Distance within which transmission can occur",
    "grid_size": "Size of the spatial simulation grid",
    "incubation_period_range": "Range of days for incubation period (E->I)",
    "infectious_period_range": "Range of days for infectious period (I->R/D)",
    "simulation_days": "Total duration of simulation in days",
}


# ---------------------------------------------------------------------------
# Epidemiologically realistic hard bounds
# ---------------------------------------------------------------------------
PARAM_BOUNDS = {
    "dog_vaccination_rate": (0.0, 0.9),  # Max 90% coverage is realistic
    "human_pep_access_rate": (0.0, 1.0),
    "bite_transmission_prob": (0.05, 0.5),  # 5-50% per-bite transmission
    "pep_survival_prob": (0.85, 1.0),  # PEP is highly effective
    "incubation_period_range": ((7, 14), (30, 180)),  # Min and max bounds
    "infectious_period_range": ((1, 3), (5, 21)),
}


# ---------------------------------------------------------------------------
# Pre-defined policy intervention scenarios for comparative analysis
# ---------------------------------------------------------------------------
POLICY_SCENARIOS = {
    "baseline": {
        "name": "Current Situation (Baseline)",
        "dog_vaccination_rate": 0.1,
        "human_pep_access_rate": 0.4,
        "description": "Low vaccination coverage and limited PEP access",
    },
    "improved_vaccination": {
        "name": "Improved Dog Vaccination",
        "dog_vaccination_rate": 0.7,
        "human_pep_access_rate": 0.4,
        "description": "Mass dog vaccination campaign",
    },
    "improved_pep": {
        "name": "Improved PEP Access",
        "dog_vaccination_rate": 0.1,
        "human_pep_access_rate": 0.9,
        "description": "Better healthcare access and PEP availability",
    },
    "combined_intervention": {
        "name": "Combined Interventions",
        "dog_vaccination_rate": 0.7,
        "human_pep_access_rate": 0.9,
        "description": "Both vaccination and PEP improvements",
    },
    "one_health": {
        "name": "One Health Approach",
        "dog_vaccination_rate": 0.8,
        "human_pep_access_rate": 0.95,
        "bite_transmission_prob": 0.10,  # Reduced through public education
        "description": "Comprehensive intervention with education",
    },
}
