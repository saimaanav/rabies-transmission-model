"""
utils.py — Visualization, Reporting, and Data Export Utilities
==============================================================

This module provides all plotting, data-export, and report-generation
functions consumed by the Streamlit dashboard (``app.py``).

Visualization Functions:
    create_seird_plot              -- Multi-trace line chart of SEIRD compartments.
    create_daily_infections_plot   -- Epidemic curve bar chart (daily new cases).
    create_spatial_heatmap         -- 2D density heatmap of agent distributions.
    create_intervention_comparison_plot -- Side-by-side scenario comparison.

Reporting Functions:
    generate_summary_report        -- Full-text policy briefing for decision-makers.

Data Export Functions:
    export_daily_data              -- Write daily simulation data to CSV.

Analysis Functions:
    calculate_r_effective          -- Simplified R_eff estimation over time.

All Plotly figures use a consistent professional colour scheme and are
configured for both interactive exploration and static image export
(via Kaleido).

Dependencies:
    - plotly: Interactive charting (go.Figure, go.Scatter, go.Bar, go.Heatmap).
    - pandas: DataFrame manipulation for time-series data.
    - numpy:  Numerical operations (binning, differentiation).
    - datetime: Timestamp generation for reports and filenames.
"""

import numpy as np
import plotly.graph_objects as go
import datetime


def create_seird_plot(data, agent_type="total", title=None):
    """Create enhanced SEIRD curves plot for specified agent type."""

    # Professional color scheme with better contrast
    colors = {
        "Susceptible": "#2C3E50",  # Dark blue-gray
        "Exposed": "#F39C12",  # Orange
        "Infectious": "#E74C3C",  # Red
        "Recovered": "#27AE60",  # Green
        "Dead": "#8E44AD",  # Purple
    }

    if agent_type == "dogs":
        cols = [
            "dogs_susceptible",
            "dogs_exposed",
            "dogs_infectious",
            "dogs_recovered",
            "dogs_dead",
        ]
    elif agent_type == "humans":
        cols = [
            "humans_susceptible",
            "humans_exposed",
            "humans_infectious",
            "humans_recovered",
            "humans_dead",
        ]
    else:  # total
        cols = [
            "total_susceptible",
            "total_exposed",
            "total_infectious",
            "total_recovered",
            "total_dead",
        ]

    labels = ["Susceptible", "Exposed", "Infectious", "Recovered", "Dead"]

    fig = go.Figure()

    # Add traces with enhanced styling
    for i, (col, label) in enumerate(zip(cols, labels)):
        if col in data.columns:
            # Use different line styles for better distinction
            line_style = "solid"
            if label == "Exposed":
                line_style = "dash"
            elif label == "Dead":
                line_style = "dot"

            fig.add_trace(
                go.Scatter(
                    x=data["day"],
                    y=data[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colors[label], width=3, dash=line_style),
                    marker=dict(
                        size=4, color=colors[label], line=dict(width=1, color="white")
                    ),
                    hovertemplate=f"<b>{label}</b><br>Day: %{{x}}<br>Count: %{{y:,}}<extra></extra>",
                )
            )

    # Enhanced layout with professional styling
    fig.update_layout(
        title=dict(
            text=title or f"SEIRD Disease Progression - {agent_type.title()}",
            font=dict(size=18, color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text="Days Since Simulation Start", font=dict(size=14, color="#2C3E50")
            ),
            tickfont=dict(size=12),
            gridcolor="#ECF0F1",
            linecolor="#BDC3C7",
        ),
        yaxis=dict(
            title=dict(text="Number of Agents", font=dict(size=14, color="#2C3E50")),
            tickfont=dict(size=12),
            gridcolor="#ECF0F1",
            linecolor="#BDC3C7",
        ),
        hovermode="x unified",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#BDC3C7",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=60, t=80, b=140),
        height=550,
    )

    return fig


def create_daily_infections_plot(data):
    """Create epidemic curve showing daily new infections with bar chart style."""

    # Calculate daily new infections correctly - use total ever infected difference
    if "dogs_ever_infected" in data.columns and "humans_ever_infected" in data.columns:
        daily_new_dogs = (
            data["dogs_ever_infected"]
            .diff()
            .fillna(data["dogs_ever_infected"].iloc[0] if len(data) > 0 else 0)
        )
        daily_new_humans = (
            data["humans_ever_infected"]
            .diff()
            .fillna(data["humans_ever_infected"].iloc[0] if len(data) > 0 else 0)
        )
    else:
        # Fallback to exposed cases
        daily_new_dogs = data["dogs_exposed"].diff().fillna(0)
        daily_new_humans = data["humans_exposed"].diff().fillna(0)

    daily_total = daily_new_dogs + daily_new_humans

    # Create single bar chart matching the screenshot style
    fig = go.Figure()

    # Add single bar trace for total new infections matching screenshot
    fig.add_trace(
        go.Bar(
            x=data["day"],
            y=daily_total,
            name="New Infections",
            marker_color="#FF8C00",  # Orange color matching screenshot
            opacity=0.9,
            text=daily_total.astype(int),
            textposition="outside",
            textfont=dict(size=10, color="black"),
            showlegend=False,
        )
    )

    # Update layout to match screenshot style
    fig.update_layout(
        title=dict(
            text="Epidemic Curve: New Infections Per Day",
            font=dict(size=16, color="black"),
            x=0.5,
        ),
        xaxis=dict(
            title="Day",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
            range=[0, max(100, data["day"].max())],
        ),
        yaxis=dict(
            title="New Human Infections",
            showgrid=True,
            gridcolor="lightgray",
            zeroline=False,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        template="plotly_white",
        height=400,
        showlegend=False,
        margin=dict(l=60, r=40, t=60, b=60),
        barmode="group",
    )

    return fig


def create_spatial_heatmap(model, metric="infections"):
    """Create spatial heatmap of infections or other metrics."""
    import numpy as np

    # Create grid for spatial visualization
    grid_size = model.grid_size
    resolution = 20  # Grid resolution for heatmap
    x_bins = np.linspace(0, grid_size, resolution)
    y_bins = np.linspace(0, grid_size, resolution)

    # Initialize heat map data
    heat_data = np.zeros((resolution - 1, resolution - 1))

    # Count agents in each grid cell
    for agent in model.all_agents:
        if hasattr(agent, "pos") and agent.pos:
            x, y = agent.pos
            # Find which bin this agent belongs to
            x_idx = np.digitize(x, x_bins) - 1
            y_idx = np.digitize(y, y_bins) - 1

            # Ensure indices are within bounds
            if 0 <= x_idx < resolution - 1 and 0 <= y_idx < resolution - 1:
                if metric == "infections" and hasattr(agent, "health_state"):
                    # Include all infected states: Exposed, Infectious, and ever infected
                    if agent.health_state in [
                        "E",
                        "I",
                        "R",
                        "D",
                    ]:  # Anyone who has been infected
                        heat_data[y_idx, x_idx] += 1
                elif metric == "total":
                    heat_data[y_idx, x_idx] += 1
                elif metric == "dogs" and hasattr(agent, "unique_id"):
                    # Check if it's a dog agent by looking at the agent type
                    if (
                        hasattr(agent, "__class__")
                        and "Dog" in agent.__class__.__name__
                    ):
                        heat_data[y_idx, x_idx] += 1
                elif metric == "humans" and hasattr(agent, "unique_id"):
                    # Check if it's a human agent by looking at the agent type
                    if (
                        hasattr(agent, "__class__")
                        and "Human" in agent.__class__.__name__
                    ):
                        heat_data[y_idx, x_idx] += 1

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heat_data,
            x=x_bins[:-1],
            y=y_bins[:-1],
            colorscale="Reds" if metric == "infections" else "Viridis",
            colorbar=dict(
                title=dict(text=f"{metric.title()} Density", font=dict(size=14))
            ),
            hoverongaps=False,
            hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Count: %{z}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(
            text=f"Spatial Distribution - {metric.title()}",
            font=dict(size=18, color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(text="X Coordinate", font=dict(size=14, color="#2C3E50"))
        ),
        yaxis=dict(
            title=dict(text="Y Coordinate", font=dict(size=14, color="#2C3E50"))
        ),
        template="plotly_white",
        height=500,
        width=600,
    )

    return fig


def generate_summary_report(data, model):
    """Generate a comprehensive summary report for policy makers."""

    final_day = data.iloc[-1]
    peak_day = data["daily_new_infections"].idxmax()
    peak_infections = data["daily_new_infections"].max()

    total_population = model.num_dogs + model.num_humans

    report = f"""
RABIES TRANSMISSION SIMULATION - POLICY SUMMARY REPORT
Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

===============================================================================
SIMULATION PARAMETERS
===============================================================================

Population:
- Dogs: {model.num_dogs:,}
- Humans: {model.num_humans:,}
- Total Population: {total_population:,}

Initial Conditions:
- Initially Infected Dogs: {model.initial_infected_dogs}
- Initially Infected Humans: {model.initial_infected_humans}

Intervention Measures:
- Dog Vaccination Rate: {model.dog_vaccination_rate * 100:.1f}%
- Human PEP Access Rate: {model.human_pep_access_rate * 100:.1f}%
- PEP Survival Probability: {model.pep_survival_prob * 100:.1f}%

Transmission Parameters:
- Bite Transmission Probability: {model.bite_transmission_prob * 100:.1f}%
- Dog Roaming Radius: {model.dog_roaming_radius:.1f} units
- Bite Contact Radius: {model.bite_contact_radius:.1f} units

Disease Parameters:
- Incubation Period: {model.incubation_period_range[0]}-{model.incubation_period_range[1]} days
- Infectious Period: {model.infectious_period_range[0]}-{model.infectious_period_range[1]} days

Simulation Duration: {model.simulation_days} days

===============================================================================
EPIDEMIC OUTCOMES
===============================================================================

Overall Impact:
- Total Ever Infected: {int(final_day["total_ever_infected"]):,} ({final_day["total_ever_infected"] / total_population * 100:.1f}% of population)
- Total Deaths: {int(final_day["total_deaths"]):,} ({final_day["total_deaths"] / total_population * 100:.2f}% of population)
- Attack Rate: {final_day["attack_rate"] * 100:.1f}%

Peak Epidemic:
- Peak Daily Infections: {int(peak_infections)} (Day {peak_day + 1})

Final Population Status:
Dogs:
- Susceptible: {int(final_day["dogs_susceptible"]):,} ({final_day["dogs_susceptible"] / model.num_dogs * 100:.1f}%)
- Dead: {int(final_day["dogs_dead"]):,} ({final_day["dogs_dead"] / model.num_dogs * 100:.1f}%)
- Ever Infected: {int(final_day["dogs_ever_infected"]):,} ({final_day["dogs_ever_infected"] / model.num_dogs * 100:.1f}%)

Humans:
- Susceptible: {int(final_day["humans_susceptible"]):,} ({final_day["humans_susceptible"] / model.num_humans * 100:.1f}%)
- Recovered: {int(final_day["humans_recovered"]):,} ({final_day["humans_recovered"] / model.num_humans * 100:.1f}%)
- Dead: {int(final_day["humans_dead"]):,} ({final_day["humans_dead"] / model.num_humans * 100:.1f}%)
- Ever Infected: {int(final_day["humans_ever_infected"]):,} ({final_day["humans_ever_infected"] / model.num_humans * 100:.1f}%)

Case Fatality Rates:
- Dogs: {final_day["dogs_dead"] / max(final_day["dogs_ever_infected"], 1) * 100:.1f}% (rabies is always fatal in dogs)
- Humans: {final_day["humans_dead"] / max(final_day["humans_ever_infected"], 1) * 100:.1f}%

===============================================================================
POLICY IMPLICATIONS
===============================================================================

Intervention Effectiveness:
"""

    # Add intervention-specific insights
    if model.dog_vaccination_rate >= 0.7:
        report += "\n✓ HIGH dog vaccination rate (≥70%) - Excellent intervention level"
    elif model.dog_vaccination_rate >= 0.5:
        report += "\n• MODERATE dog vaccination rate (50-69%) - Good intervention level"
    else:
        report += "\n⚠ LOW dog vaccination rate (<50%) - Insufficient intervention"

    if model.human_pep_access_rate >= 0.8:
        report += "\n✓ HIGH PEP access rate (≥80%) - Excellent healthcare access"
    elif model.human_pep_access_rate >= 0.6:
        report += "\n• MODERATE PEP access rate (60-79%) - Good healthcare access"
    else:
        report += "\n⚠ LOW PEP access rate (<60%) - Insufficient healthcare access"

    report += """

Key Recommendations:
"""

    # Generate recommendations based on outcomes
    if final_day["attack_rate"] > 0.1:  # High attack rate
        report += "\n1. URGENT: Implement mass dog vaccination campaigns (target >70% coverage)"
        report += "\n2. CRITICAL: Improve PEP accessibility and availability in affected areas"
        report += "\n3. Enhance surveillance and rapid response capabilities"
    elif final_day["attack_rate"] > 0.05:  # Moderate attack rate
        report += "\n1. Strengthen dog vaccination programs in high-risk areas"
        report += "\n2. Improve public awareness about rabies prevention"
        report += "\n3. Ensure adequate PEP supplies and training"
    else:  # Low attack rate
        report += "\n1. Maintain current vaccination levels"
        report += "\n2. Continue surveillance and monitoring"
        report += "\n3. Prepare for potential outbreaks"

    if final_day["humans_dead"] > 0:
        report += f"\n4. PRIORITY: Address human deaths ({int(final_day['humans_dead'])} deaths preventable with better PEP access)"

    report += f"""

Economic Considerations:
- Prevention Cost vs Treatment Cost: Mass vaccination is typically 10-15x more cost-effective than post-exposure treatment
- Human Lives at Risk: {int(final_day["humans_ever_infected"])} people were infected in this scenario
- Preventable Deaths: {int(final_day["humans_dead"])} human deaths could have been prevented with better PEP access

===============================================================================
TECHNICAL NOTES
===============================================================================

Model Assumptions:
- Spatial transmission based on proximity within {model.bite_contact_radius} units
- Stochastic disease progression with random incubation/infectious periods
- Dogs always die from rabies; humans can recover with PEP treatment
- Limited human mobility compared to dog roaming behavior

Limitations:
- Model does not account for seasonal variations
- Simplified spatial structure (uniform grid)
- Does not include wildlife reservoirs
- Population is closed (no births/deaths except from rabies)

Data Quality: This simulation provides relative comparisons between intervention scenarios.
Absolute numbers should be interpreted with caution and validated against field data.

===============================================================================
"""

    return report


def export_daily_data(data, filename=None):
    """Export daily simulation data to CSV."""
    if filename is None:
        filename = (
            f"rabies_simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

    data.to_csv(filename, index=False)
    return filename


def calculate_r_effective(data, generation_time=30):
    """Calculate effective reproduction number (R_eff) over time."""
    # Simplified calculation based on growth rate
    # In practice, this would require more sophisticated epidemiological analysis

    r_eff = []
    window = generation_time

    for i in range(len(data)):
        if i < window:
            r_eff.append(np.nan)
        else:
            current_cases = data.iloc[i]["total_infectious"]
            previous_cases = data.iloc[i - window]["total_infectious"]

            if previous_cases > 0:
                growth_rate = current_cases / previous_cases
                r_eff.append(growth_rate)
            else:
                r_eff.append(np.nan)

    return r_eff


def create_intervention_comparison_plot(scenarios_data):
    """Create comparison plot for multiple intervention scenarios."""

    fig = go.Figure()

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12", "#9B59B6"]

    for i, (scenario_name, data) in enumerate(scenarios_data.items()):
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["total_ever_infected"],
                mode="lines",
                name=scenario_name,
                line=dict(width=3, color=colors[i % len(colors)]),
                hovertemplate=f"<b>{scenario_name}</b><br>"
                + "Day: %{x}<br>"
                + "Total Infections: %{y:,.0f}<br>"
                + "<extra></extra>",
            )
        )

    fig.update_layout(
        title=dict(
            text="Intervention Scenario Comparison - Total Infections Over Time",
            font=dict(size=18, color="#2C3E50"),
            x=0.5,
        ),
        xaxis=dict(
            title=dict(
                text="Days Since Outbreak Start", font=dict(size=14, color="#2C3E50")
            ),
            gridcolor="#E8E8E8",
        ),
        yaxis=dict(
            title=dict(
                text="Cumulative Infections", font=dict(size=14, color="#2C3E50")
            ),
            gridcolor="#E8E8E8",
        ),
        template="plotly_white",
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="x unified",
    )

    return fig
