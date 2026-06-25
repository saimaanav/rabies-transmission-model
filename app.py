"""
app.py — Streamlit Interactive Dashboard for the Rabies Transmission Model
===========================================================================

Entry point for the interactive web dashboard. Provides sidebar parameter
controls, simulation execution with progress tracking, five-tab result
visualization (SEIRD curves, epidemic curves, spatial heatmaps, raw data
export, and automated policy summary reports), plus a comprehensive
epidemiological terms dictionary.

Functions:
    main()                    -- Dashboard entry point and simulation controller.
    show_dictionary_content() -- Renders the terms & definitions glossary.
    display_results()         -- Renders all five result tabs with charts.

Usage:
    ``streamlit run app.py``
"""

import streamlit as st
import pandas as pd
import io
from model import RabiesModel
from params import DEFAULT_PARAMS
from utils import (
    generate_summary_report,
    create_seird_plot,
    create_daily_infections_plot,
    create_spatial_heatmap,
)
from parameter_tuning import DELHI_PARAMS

st.set_page_config(
    page_title="Rabies Transmission Simulator",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS — Dark-themed styling with animations
# ---------------------------------------------------------------------------
st.markdown(
    """<style>
:root{--primary-color:#667eea;--secondary-color:#764ba2;--accent-color:#f093fb;--success-color:#2ecc71;--warning-color:#f39c12;--danger-color:#e74c3c;--dark-bg:#0a0a0a;--card-bg:rgba(255,255,255,0.05);--text-color:#fff;--border-color:rgba(255,255,255,0.1)}
.stApp{background:linear-gradient(135deg,#0a0a0a 0%,#1a1a1a 50%,#0f0f0f 100%);color:var(--text-color)}
.main .block-container{background:transparent;color:var(--text-color);animation:fadeIn 1s ease-out}
.main-header{background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));padding:2rem;border-radius:20px;margin-bottom:2rem;color:#fff;text-align:center;box-shadow:0 8px 32px rgba(102,126,234,.3);animation:slideInDown 1s ease-out;border:1px solid var(--border-color)}
.main-header h1{margin:0;font-size:2.5rem;font-weight:600;animation:glow 2s ease-in-out infinite alternate}
.main-header p{margin:.5rem 0;opacity:.9}
.simulation-controls{background:linear-gradient(135deg,rgba(102,126,234,.1),rgba(118,75,162,.1));padding:2.5rem;border-radius:20px;border:1px solid var(--border-color);text-align:center;margin:2rem 0;animation:fadeInScale 1s ease-out;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(102,126,234,.2)}
.dictionary-section{background:var(--card-bg);padding:2rem;border-radius:15px;border:1px solid var(--border-color);margin:2rem 0;animation:slideInUp .8s ease-out;backdrop-filter:blur(10px);box-shadow:0 8px 32px rgba(255,255,255,.05)}
.metric-card{background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));padding:1.5rem;border-radius:15px;box-shadow:0 8px 32px rgba(102,126,234,.4);color:#fff;text-align:center;margin-bottom:1rem;animation:slideInUp .8s ease-out;border:1px solid var(--border-color);transition:transform .3s ease,box-shadow .3s ease}
.metric-card:hover{transform:translateY(-5px);box-shadow:0 12px 40px rgba(102,126,234,.6)}
.chart-container{background:var(--card-bg);padding:1.5rem;border-radius:15px;box-shadow:0 8px 32px rgba(255,255,255,.05);backdrop-filter:blur(10px);border:1px solid var(--border-color);margin-bottom:2rem;animation:fadeInScale .8s ease-out}
.section-divider{height:2px;background:linear-gradient(90deg,transparent,var(--primary-color),var(--secondary-color),transparent);border-radius:1px;margin:2rem 0}
.acknowledgements{text-align:center;font-size:.75rem;color:#ccc;margin-top:3rem;padding:1.5rem;border-top:1px solid var(--border-color);background:rgba(255,255,255,.02);border-radius:10px}
.stButton>button{background:linear-gradient(135deg,var(--secondary-color),var(--accent-color));color:#fff;border:none;border-radius:25px;padding:.75rem 2rem;font-weight:600;font-size:1.1rem;transition:all .3s ease;box-shadow:0 3px 10px rgba(52,152,219,.3)}
.stButton>button:hover{transform:translateY(-2px);box-shadow:0 5px 15px rgba(52,152,219,.4)}
.stSidebar>div{background:linear-gradient(135deg,#0f0f0f,#1a1a1a);border-right:1px solid var(--border-color)}
.stSidebar .stMarkdown,.stSidebar .stMarkdown p,.stSidebar .stMarkdown h1,.stSidebar .stMarkdown h2,.stSidebar .stMarkdown h3,.stSidebar .stMarkdown h4{color:#fff!important}
.stSidebar label,.stSidebar .stSelectbox label,.stSidebar .stSlider label,.stSidebar .stNumberInput label{color:#fff!important;font-weight:500}
.stSidebar h3,.stSidebar h2,.stSidebar h1{color:#fff!important;font-weight:600!important}
.stApp .stMarkdown,.stApp .stMarkdown p,.stApp .stMarkdown div{color:#fff!important}
.stApp .stMarkdown h1,.stApp .stMarkdown h2,.stApp .stMarkdown h3,.stApp .stMarkdown h4{color:#fff!important}
.stProgress>div>div>div>div{background:linear-gradient(135deg,var(--primary-color),var(--secondary-color))}
.stTabs [data-baseweb="tab-list"]{background:var(--card-bg);border-radius:10px;border:1px solid var(--border-color)}
.stTabs [data-baseweb="tab"]{color:var(--text-color);background:transparent;transition:all .3s ease}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--primary-color),var(--secondary-color));color:#fff}
@keyframes fadeIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideInDown{from{opacity:0;transform:translateY(-30px)}to{opacity:1;transform:translateY(0)}}
@keyframes slideInUp{from{opacity:0;transform:translateY(30px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeInScale{from{opacity:0;transform:scale(.95)}to{opacity:1;transform:scale(1)}}
@keyframes glow{from{text-shadow:0 0 10px rgba(102,126,234,.5)}to{text-shadow:0 0 20px rgba(102,126,234,.8)}}
@keyframes bounceIn{0%{opacity:0;transform:scale(.3)}50%{opacity:1;transform:scale(1.05)}70%{transform:scale(.9)}100%{opacity:1;transform:scale(1)}}
</style>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------


def main():
    """Dashboard entry point: renders header, sidebar controls, simulation runner, and result display."""
    st.markdown(
        """<div class="main-header"><h1>🐕 Rabies Transmission Simulator & Policy Dashboard</h1><p><strong>Agent-Based Model for Rabies Transmission in India</strong></p><p>Interactive dashboard to simulate rabies transmission between dogs and humans and test intervention strategies in real-time.</p></div>""",
        unsafe_allow_html=True,
    )

    if "simulation_data" not in st.session_state:
        st.session_state.simulation_data = None
    if "model_results" not in st.session_state:
        st.session_state.model_results = None
    if "simulation_complete" not in st.session_state:
        st.session_state.simulation_complete = False

    # ---- Sidebar Controls ----
    st.sidebar.markdown(
        """<div class="sidebar-header"><h2>🎛️ Simulation Parameters</h2><p>Configure your rabies transmission model</p></div>""",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### 🏙️ Delhi Urban Configuration")
    delhi_info = DELHI_PARAMS.get("_spatial_info", {})
    if delhi_info:
        st.sidebar.markdown(
            f"**Spatial Representation:**\n- Grid: {DELHI_PARAMS['grid_size']}×{DELHI_PARAMS['grid_size']} units\n- Scale: {delhi_info.get('km_per_grid_unit', 1)} km per unit\n- Area: {delhi_info.get('total_area_km2', 1600):,} km²"
        )
    if st.sidebar.button("🏙️ Load Delhi Urban Configuration", use_container_width=True):
        for key, value in DELHI_PARAMS.items():
            if not key.startswith("_"):
                st.session_state[f"delhi_{key}"] = value
        st.sidebar.success("Delhi configuration loaded!")
        st.rerun()

    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### 👥 Population Settings")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        num_dogs = st.number_input(
            "Dogs",
            value=st.session_state.get("delhi_num_dogs", DEFAULT_PARAMS["num_dogs"]),
            help="Total dogs",
        )
    with col2:
        num_humans = st.number_input(
            "Humans",
            value=st.session_state.get(
                "delhi_num_humans", DEFAULT_PARAMS["num_humans"]
            ),
            help="Total humans",
        )
    initial_infected_dogs = st.sidebar.number_input(
        "Initial Infected Dogs",
        value=st.session_state.get(
            "delhi_initial_infected_dogs", DEFAULT_PARAMS["initial_infected_dogs"]
        ),
        help="Dogs infected at start",
    )
    initial_infected_humans = 0

    st.sidebar.subheader("Intervention Settings")
    dog_vaccination_rate = (
        st.sidebar.slider(
            "Dog Vaccination Rate (%)",
            0.0,
            100.0,
            st.session_state.get(
                "delhi_dog_vaccination_rate", DEFAULT_PARAMS["dog_vaccination_rate"]
            )
            * 100,
            1.0,
        )
        / 100.0
    )
    human_pep_access_rate = (
        st.sidebar.slider(
            "Human PEP Access Rate (%)",
            0.0,
            100.0,
            st.session_state.get(
                "delhi_human_pep_access_rate", DEFAULT_PARAMS["human_pep_access_rate"]
            )
            * 100,
            1.0,
        )
        / 100.0
    )

    st.sidebar.subheader("Transmission Settings")
    bite_transmission_prob = (
        st.sidebar.slider(
            "Bite Transmission Probability (%)",
            0.0,
            100.0,
            st.session_state.get(
                "delhi_bite_transmission_prob", DEFAULT_PARAMS["bite_transmission_prob"]
            )
            * 100,
            1.0,
        )
        / 100.0
    )
    pep_survival_prob = (
        st.sidebar.slider(
            "PEP Survival Probability (%)",
            0.0,
            100.0,
            st.session_state.get(
                "delhi_pep_survival_prob", DEFAULT_PARAMS["pep_survival_prob"]
            )
            * 100,
            1.0,
        )
        / 100.0
    )

    st.sidebar.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### 🗺️ Spatial Settings")
    grid_size = st.sidebar.number_input(
        "Grid Size (units)",
        value=st.session_state.get("delhi_grid_size", DEFAULT_PARAMS["grid_size"]),
    )
    col1, col2 = st.sidebar.columns(2)
    with col1:
        dog_roaming_radius = st.number_input(
            "Dog Roaming Radius",
            value=st.session_state.get(
                "delhi_dog_roaming_radius", DEFAULT_PARAMS["dog_roaming_radius"]
            ),
            step=0.5,
        )
    with col2:
        bite_contact_radius = st.number_input(
            "Bite Contact Radius",
            value=st.session_state.get(
                "delhi_bite_contact_radius", DEFAULT_PARAMS["bite_contact_radius"]
            ),
            step=0.1,
        )

    st.sidebar.subheader("Human Mobility Settings")
    human_mobility_radius = st.sidebar.number_input(
        "Human Mobility Radius",
        value=st.session_state.get(
            "delhi_human_mobility_radius", DEFAULT_PARAMS["human_mobility_radius"]
        ),
        step=0.5,
    )

    st.sidebar.subheader("Disease Parameters")
    delhi_incubation = st.session_state.get(
        "delhi_incubation_period_range", DEFAULT_PARAMS["incubation_period_range"]
    )
    delhi_infectious = st.session_state.get(
        "delhi_infectious_period_range", DEFAULT_PARAMS["infectious_period_range"]
    )
    incubation_min = st.sidebar.number_input(
        "Min Incubation Period (days)", value=delhi_incubation[0]
    )
    incubation_max = st.sidebar.number_input(
        "Max Incubation Period (days)", value=delhi_incubation[1]
    )
    infectious_min = st.sidebar.number_input(
        "Min Infectious Period (days)", value=delhi_infectious[0]
    )
    infectious_max = st.sidebar.number_input(
        "Max Infectious Period (days)", value=delhi_infectious[1]
    )

    st.sidebar.subheader("Simulation Settings")
    simulation_days = st.sidebar.number_input(
        "Simulation Duration (days)",
        value=st.session_state.get(
            "delhi_simulation_days", DEFAULT_PARAMS["simulation_days"]
        ),
    )
    st.sidebar.markdown("### 🎲 Reproducibility Control")
    use_random_seed = st.sidebar.checkbox(
        "Use Random Seed", value=st.session_state.get("delhi_random_seed") is not None
    )
    random_seed = None
    if use_random_seed:
        random_seed = st.sidebar.number_input(
            "Random Seed", value=st.session_state.get("delhi_random_seed", 42)
        )

    # ---- Main Content ----
    if not st.session_state.simulation_complete:
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(
                """<div class="simulation-controls"><h2 style="color:#fff;margin-bottom:1rem;">🚀 Ready to Run Simulation</h2><p style="color:#ccc;margin-bottom:2rem;">Configure parameters in the sidebar and click below to start</p></div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                """<div style="background:linear-gradient(135deg,#1e3a8a,#3730a3);padding:1rem;border-radius:10px;margin:1rem 0;"><h4 style="color:#fff;margin:0 0 .5rem 0;">🎲 Why Results Vary Each Run</h4><p style="color:#e5e7eb;margin:0;font-size:.9rem;">This is a stochastic simulation. Enable "Use Random Seed" in the sidebar for identical results.</p></div>""",
                unsafe_allow_html=True,
            )
            _, run_col, _ = st.columns([1, 2, 1])
            with run_col:
                run_simulation_button = st.button(
                    "🚀 Run Simulation",
                    type="primary",
                    use_container_width=True,
                    key="main_run_button",
                )
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown(
                """<div class="dictionary-section"><h2 style="color:#fff;text-align:center;margin-bottom:1.5rem;">📚 Terms & Definitions Dictionary</h2></div>""",
                unsafe_allow_html=True,
            )
            show_dictionary_content()
    else:
        _, _, col3, _, _ = st.columns([1, 1, 1, 1, 1])
        with col3:
            if st.button(
                "🔄 Reset Simulation", use_container_width=True, type="secondary"
            ):
                st.session_state.simulation_data = None
                st.session_state.model_results = None
                st.session_state.simulation_complete = False
                st.rerun()
        run_simulation_button = False

    # ---- Simulation Execution ----
    if run_simulation_button if not st.session_state.simulation_complete else False:
        params = {
            "num_dogs": num_dogs,
            "num_humans": num_humans,
            "initial_infected_dogs": initial_infected_dogs,
            "initial_infected_humans": initial_infected_humans,
            "dog_vaccination_rate": dog_vaccination_rate,
            "human_pep_access_rate": human_pep_access_rate,
            "bite_transmission_prob": bite_transmission_prob,
            "pep_survival_prob": pep_survival_prob,
            "dog_roaming_radius": dog_roaming_radius,
            "bite_contact_radius": bite_contact_radius,
            "grid_size": grid_size,
            "incubation_period_range": (incubation_min, incubation_max),
            "infectious_period_range": (infectious_min, infectious_max),
            "simulation_days": simulation_days,
            "human_mobility_radius": human_mobility_radius,
            "random_seed": random_seed,
        }
        total_population = params["num_dogs"] + params["num_humans"]
        if total_population > 100000:
            st.warning(
                f"⚠️ Large Population ({total_population:,}). Statistical sampling will be used."
            )
        progress_bar = st.progress(0)
        status_text = st.empty()
        try:
            status_text.text("Initializing simulation...")
            model = RabiesModel(**params)
            daily_data = []
            for step in range(simulation_days):
                progress_bar.progress((step + 1) / simulation_days)
                status_text.text(f"Day {step + 1}/{simulation_days}")
                model.step()
                daily_stats = model.get_daily_statistics()
                daily_stats["day"] = step + 1
                daily_data.append(daily_stats)
            st.session_state.simulation_data = pd.DataFrame(daily_data)
            st.session_state.model_results = model
            st.session_state.simulation_complete = True
            progress_bar.progress(1.0)
            status_text.text("✅ Simulation completed!")
            st.rerun()
        except MemoryError:
            st.error("❌ Memory Error. Reduce population sizes.")
            progress_bar.empty()
            status_text.empty()
        except Exception as e:
            st.error(f"Simulation failed: {str(e)}")
            progress_bar.empty()
            status_text.empty()

    if (
        st.session_state.simulation_complete
        and st.session_state.simulation_data is not None
    ):
        display_results(
            st.session_state.simulation_data, st.session_state.model_results
        )


# ---------------------------------------------------------------------------
# Terms & Definitions Dictionary
# ---------------------------------------------------------------------------


def show_dictionary_content():
    """Display comprehensive dictionary of epidemiological and modelling terms."""
    st.subheader("🦠 Disease Model (SEIRD)")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "**Susceptible (S)**: Can catch rabies but not yet infected.\n\n**Exposed (E)**: Infected but not yet symptomatic/transmitting.\n\n**Infectious (I)**: Actively symptomatic and can transmit via bites."
        )
    with col2:
        st.markdown(
            "**Recovered (R)**: Survived infection (very rare without treatment).\n\n**Dead (D)**: Died from rabies.\n\n**Ever Infected**: Total agents in E, I, R, or D at any point."
        )
    st.subheader("⚙️ Parameters")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown(
            "**Dog Vaccination Rate**: % of dogs vaccinated.\n\n**PEP Access Rate**: % of humans with PEP access.\n\n**Bite Transmission Prob**: Chance a bite transmits rabies.\n\n**PEP Survival Prob**: Treatment success rate."
        )
    with col4:
        st.markdown(
            "**Dog Roaming Radius**: Daily dog movement distance.\n\n**Bite Contact Radius**: Distance for bite to occur.\n\n**Human Mobility Radius**: Daily human travel distance.\n\n**Incubation Period**: Days E→I. **Infectious Period**: Days I→outcome."
        )
    st.subheader("🏥 Medical Terms")
    st.markdown(
        "**PEP**: Post-exposure prophylaxis (vaccine series after bite). **Rabies**: Fatal viral neurological disease. **Zoonosis**: Animal-to-human disease. **Reservoir Host**: Species that harbors pathogen (dogs for rabies). **Herd Immunity**: Population-level protection via vaccination."
    )
    st.subheader("🔬 Modelling Terms")
    st.markdown(
        "**ABM**: Agent-based model. **Stochastic**: Includes randomness. **Compartmental**: Population divided by health state. **Continuous Space**: Agents move freely (not grid-locked). **Monte Carlo**: Repeated random sampling method."
    )


# ---------------------------------------------------------------------------
# Result Display
# ---------------------------------------------------------------------------


def display_results(data, model):
    """Display simulation results across five interactive tabs with charts and exports."""
    st.markdown(
        """<div style="background:linear-gradient(135deg,#2ecc71,#27ae60);color:#fff;padding:1rem;border-radius:10px;text-align:center;margin:1rem 0;">✅ Simulation Complete!</div>""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div class="chart-container"><h1 style="text-align:center;color:#2C3E50;">📊 Simulation Results & Analysis</h1></div>""",
        unsafe_allow_html=True,
    )

    final_day = data.iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3 style="color:#E74C3C;margin:0;">Total Infections</h3><h2 style="margin:.5rem 0;">{int(final_day["total_ever_infected"]):,}</h2></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3 style="color:#8E44AD;margin:0;">Total Deaths</h3><h2 style="margin:.5rem 0;">{int(final_day["total_deaths"]):,}</h2></div>',
            unsafe_allow_html=True,
        )
    with col3:
        peak = (
            int(data["daily_new_infections"].max())
            if "daily_new_infections" in data.columns
            else 0
        )
        pday = (
            int(data["daily_new_infections"].idxmax() + 1)
            if "daily_new_infections" in data.columns
            else 0
        )
        st.markdown(
            f'<div class="metric-card"><h3 style="color:#F39C12;margin:0;">Peak Daily Cases</h3><h2 style="margin:.5rem 0;">{peak:,}</h2><p style="color:#7F8C8D;margin:0;">Day {pday}</p></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 SEIRD", "📊 Epi Curves", "🗺️ Spatial", "📋 Data", "📄 Policy"]
    )

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_seird_plot(data, "dogs")
            st.plotly_chart(fig, use_container_width=True, key="seird_dogs")
        with col2:
            fig = create_seird_plot(data, "humans")
            st.plotly_chart(fig, use_container_width=True, key="seird_humans")
        fig = create_seird_plot(data, "total")
        st.plotly_chart(fig, use_container_width=True, key="seird_total")

    with tab2:
        fig = create_daily_infections_plot(data)
        st.plotly_chart(fig, use_container_width=True, key="daily_infections")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_spatial_heatmap(model, "infections")
            st.plotly_chart(fig, use_container_width=True, key="hm_infections")
        with col2:
            fig = create_spatial_heatmap(model, "total")
            st.plotly_chart(fig, use_container_width=True, key="hm_total")
        col3, col4 = st.columns(2)
        with col3:
            fig = create_spatial_heatmap(model, "dogs")
            st.plotly_chart(fig, use_container_width=True, key="hm_dogs")
        with col4:
            fig = create_spatial_heatmap(model, "humans")
            st.plotly_chart(fig, use_container_width=True, key="hm_humans")

    with tab4:
        st.dataframe(data, use_container_width=True)
        buf = io.StringIO()
        data.to_csv(buf, index=False)
        st.download_button(
            "📥 Download CSV",
            buf.getvalue(),
            f"rabies_data_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True,
        )

    with tab5:
        try:
            report = generate_summary_report(data, model)
            st.text_area("Policy Summary", report, height=400)
            st.download_button(
                "📄 Download Report",
                report,
                f"rabies_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.txt",
                "text/plain",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Report generation failed: {e}")

    st.markdown(
        """<div class="acknowledgements"><p><strong>Model Created By:</strong> Maanav Chittireddy</p><p><strong>Supported By:</strong> Non Trivial</p></div>""",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
