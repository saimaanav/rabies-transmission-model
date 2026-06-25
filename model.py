"""
model.py — Core Agent-Based Model for Rabies Transmission
==========================================================

This module implements the ``RabiesModel`` class, which orchestrates the
entire rabies transmission simulation. It is responsible for:

    - Initialising the spatial environment (Mesa ``ContinuousSpace``).
    - Creating and placing dog and human agent populations.
    - Executing the simulation loop (daily time steps).
    - Collecting epidemiological statistics via Mesa's ``DataCollector``.
    - Providing optimised neighbor-detection for large populations.

Performance Strategy:
    For populations exceeding 50,000 agents, the model employs a statistical
    sampling approach: it scales down the simulated population while
    maintaining epidemiological ratios, then scales results back up for
    reporting.  This enables simulation of city-scale populations (e.g.,
    Delhi with 35M+ inhabitants) without prohibitive memory or CPU costs.

Design Patterns:
    - **Facade**: ``RabiesModel`` presents a simple ``step()`` /
      ``get_daily_statistics()`` interface while encapsulating complex
      agent management, spatial indexing, and data collection.
    - **Observer**: Mesa's ``DataCollector`` observes model state each step
      without coupling to the agent logic.

Classes:
    RabiesModel — The top-level simulation controller.

Dependencies:
    - mesa (>= 3.0): Agent, Model, ContinuousSpace, DataCollector
    - numpy: Random sampling and numerical operations
    - agents: DogAgent, HumanAgent (defined in agents.py)
"""

import numpy as np
import random
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector
from agents import DogAgent, HumanAgent


class RabiesModel(Model):
    """
    A spatially-explicit agent-based model simulating rabies transmission
    between dog and human populations.

    The model manages two agent populations (dogs and humans) placed within
    a continuous 2D space. Each simulation step represents one day, during
    which agents move, attempt transmission, and progress through the SEIRD
    disease states.

    Attributes:
        num_dogs (int):                  Original (unscaled) dog population size.
        num_humans (int):                Original (unscaled) human population size.
        initial_infected_dogs (int):     Number of dogs infected at t=0.
        initial_infected_humans (int):   Number of humans infected at t=0.
        dog_vaccination_rate (float):    Fraction of dogs vaccinated [0, 1].
        human_pep_access_rate (float):   Fraction of humans with PEP access [0, 1].
        bite_transmission_prob (float):  Per-bite transmission probability [0, 1].
        pep_survival_prob (float):       PEP treatment success probability [0, 1].
        dog_roaming_radius (float):      Maximum daily dog movement distance.
        bite_contact_radius (float):     Radius within which bites can occur.
        grid_size (int):                 Side length of the square simulation grid.
        incubation_period_range (tuple): (min, max) days for E -> I transition.
        infectious_period_range (tuple): (min, max) days for I -> R/D transition.
        simulation_days (int):           Total simulation duration in days.
        human_mobility_radius (float):   Maximum daily human movement distance.
        scale_factor (float):            Population downscaling ratio (1.0 = no scaling).
        day (int):                       Current simulation day counter.
        all_agents (list):               Master list of all agent instances.
        space (ContinuousSpace):         Mesa continuous 2D spatial environment.
        datacollector (DataCollector):   Mesa data collection instrument.

    Performance Notes:
        - Populations > 100,000 trigger adaptive grid sizing (grid_size = 100).
        - Populations > 50,000 trigger statistical sampling (scale_factor < 1.0).
        - Neighbor detection uses spatial sampling for populations > 10,000.
    """

    def __init__(
        self,
        num_dogs=100,
        num_humans=500,
        initial_infected_dogs=5,
        initial_infected_humans=0,
        dog_vaccination_rate=0.1,
        human_pep_access_rate=0.7,
        bite_transmission_prob=0.15,
        pep_survival_prob=0.99,
        dog_roaming_radius=5.0,
        bite_contact_radius=1.0,
        grid_size=100,
        incubation_period_range=(14, 60),
        infectious_period_range=(3, 10),
        simulation_days=365,
        human_mobility_radius=2.0,
        random_seed=None,
    ):
        """
        Initialise the RabiesModel with the given epidemiological parameters.

        Args:
            num_dogs (int):                  Total dog population.
            num_humans (int):                Total human population.
            initial_infected_dogs (int):     Dogs infected at t=0.
            initial_infected_humans (int):   Humans infected at t=0.
            dog_vaccination_rate (float):    Vaccination coverage [0, 1].
            human_pep_access_rate (float):   PEP access fraction [0, 1].
            bite_transmission_prob (float):  Per-bite transmission prob [0, 1].
            pep_survival_prob (float):       PEP survival prob [0, 1].
            dog_roaming_radius (float):      Dog movement radius (grid units).
            bite_contact_radius (float):     Bite interaction radius.
            grid_size (int):                 Square grid side length.
            incubation_period_range (tuple): (min_days, max_days) for E state.
            infectious_period_range (tuple): (min_days, max_days) for I state.
            simulation_days (int):           Duration of simulation.
            human_mobility_radius (float):   Human movement radius.
            random_seed (int or None):       Seed for reproducibility.
        """
        super().__init__()

        # ---- Reproducibility: seed all RNGs if requested ------------------
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
            self.random.seed(random_seed)

        # ---- Store all parameters as instance attributes ------------------
        self.num_dogs = num_dogs
        self.num_humans = num_humans
        self.initial_infected_dogs = initial_infected_dogs
        self.initial_infected_humans = initial_infected_humans
        self.dog_vaccination_rate = dog_vaccination_rate
        self.human_pep_access_rate = human_pep_access_rate
        self.bite_transmission_prob = bite_transmission_prob
        self.pep_survival_prob = pep_survival_prob
        self.dog_roaming_radius = dog_roaming_radius
        self.bite_contact_radius = bite_contact_radius
        self.grid_size = grid_size
        self.incubation_period_range = incubation_period_range
        self.infectious_period_range = infectious_period_range
        self.simulation_days = simulation_days
        self.human_mobility_radius = human_mobility_radius

        # ---- Adaptive grid sizing for very large populations --------------
        # Shrinking the grid increases agent density, which is necessary to
        # maintain realistic contact rates when populations are huge.
        total_agents = num_dogs + num_humans
        if total_agents > 100000:
            self.grid_size = 100  # Compact grid for maximum density
        else:
            self.grid_size = grid_size

        # ---- Initialise spatial environment and agent container -----------
        self.space = ContinuousSpace(self.grid_size, self.grid_size, True)
        self.all_agents = []  # Custom list (Mesa 3.0+ uses .agents internally)

        # ---- Population scaling for computational tractability ------------
        # For very large populations (> 50k), we simulate a representative
        # sample and scale results back up for reporting.
        self.scale_factor = 1.0
        if total_agents > 50000:
            self.scale_factor = min(50000 / total_agents, 1.0)
            # Ensure minimum viable populations for transmission dynamics
            self.scaled_num_dogs = max(int(num_dogs * self.scale_factor), 2000)
            self.scaled_num_humans = max(int(num_humans * self.scale_factor), 8000)
            # Scale initial infected proportionally, with a floor of 5%
            self.scaled_initial_infected_dogs = max(
                int(initial_infected_dogs * self.scale_factor),
                max(50, int(self.scaled_num_dogs * 0.05)),
            )
        else:
            self.scaled_num_dogs = num_dogs
            self.scaled_num_humans = num_humans
            self.scaled_initial_infected_dogs = initial_infected_dogs

        # ---- Simulation state tracking ------------------------------------
        self.day = 0
        self.dogs_ever_infected = 0  # Cumulative dog infections
        self.humans_ever_infected = 0  # Cumulative human infections

        # ---- Create agent populations -------------------------------------
        self._create_agents()

        # ---- Efficient state counters (avoid full iteration each step) ----
        self.dogs_s = self.scaled_num_dogs - self.scaled_initial_infected_dogs
        self.dogs_e = 0
        self.dogs_i = self.scaled_initial_infected_dogs
        self.dogs_r = 0
        self.dogs_d = 0

        self.humans_s = self.scaled_num_humans - (
            self.initial_infected_humans
            if hasattr(self, "initial_infected_humans")
            else 0
        )
        self.humans_e = 0
        self.humans_i = (
            self.initial_infected_humans
            if hasattr(self, "initial_infected_humans")
            else 0
        )
        self.humans_r = 0
        self.humans_d = 0

        # ---- Data collection via Mesa's DataCollector ---------------------
        # Lambda reporters count actual agents and scale back to original
        # population size for accurate epidemiological reporting.
        self.datacollector = DataCollector(
            model_reporters={
                "Dogs_Susceptible": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, DogAgent) and a.health_state == "S"
                    )
                    / m.scale_factor
                ),
                "Dogs_Exposed": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, DogAgent) and a.health_state == "E"
                    )
                    / m.scale_factor
                ),
                "Dogs_Infectious": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, DogAgent) and a.health_state == "I"
                    )
                    / m.scale_factor
                ),
                "Dogs_Recovered": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, DogAgent) and a.health_state == "R"
                    )
                    / m.scale_factor
                ),
                "Dogs_Dead": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, DogAgent) and a.health_state == "D"
                    )
                    / m.scale_factor
                ),
                "Humans_Susceptible": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, HumanAgent) and a.health_state == "S"
                    )
                    / m.scale_factor
                ),
                "Humans_Exposed": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, HumanAgent) and a.health_state == "E"
                    )
                    / m.scale_factor
                ),
                "Humans_Infectious": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, HumanAgent) and a.health_state == "I"
                    )
                    / m.scale_factor
                ),
                "Humans_Recovered": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, HumanAgent) and a.health_state == "R"
                    )
                    / m.scale_factor
                ),
                "Humans_Dead": lambda m: int(
                    sum(
                        1
                        for a in m.all_agents
                        if isinstance(a, HumanAgent) and a.health_state == "D"
                    )
                    / m.scale_factor
                ),
            }
        )

        # Collect initial state (day 0)
        self.datacollector.collect(self)

    def _create_agents(self):
        """
        Create and place the initial populations of dogs and humans.

        Agents are placed uniformly at random across the continuous space.
        Vaccination status and initial infection state are assigned according
        to the configured rates and counts.

        Uses the scaled population sizes when statistical sampling is active.
        """
        agent_id = 0

        # ---- Create dog agents -------------------------------------------
        for i in range(self.scaled_num_dogs):
            # Random initial position in continuous space
            x = self.random.uniform(0, self.grid_size)
            y = self.random.uniform(0, self.grid_size)

            # Determine vaccination status via Bernoulli trial
            vaccinated = self.random.random() < self.dog_vaccination_rate

            # Assign initial health state (first N dogs start Infectious)
            if i < self.scaled_initial_infected_dogs:
                health_state = "I"
                self.dogs_ever_infected += 1
            else:
                health_state = "S"

            dog = DogAgent(
                unique_id=agent_id,
                model=self,
                pos=(x, y),
                health_state=health_state,
                vaccinated=vaccinated,
                roaming_radius=self.dog_roaming_radius,
            )
            self.all_agents.append(dog)
            agent_id += 1

        # ---- Create human agents -----------------------------------------
        for i in range(self.scaled_num_humans):
            x = self.random.uniform(0, self.grid_size)
            y = self.random.uniform(0, self.grid_size)

            # Determine PEP access via Bernoulli trial
            pep_access = self.random.random() < self.human_pep_access_rate

            # All humans start with medium mobility (simplified)
            mobility_level = "medium"

            # Assign initial health state
            if i < self.initial_infected_humans:
                health_state = "I"
                self.humans_ever_infected += 1
            else:
                health_state = "S"

            human = HumanAgent(
                unique_id=agent_id,
                model=self,
                pos=(x, y),
                health_state=health_state,
                pep_access=pep_access,
                mobility_level=mobility_level,
            )
            self.all_agents.append(human)
            agent_id += 1

    def step(self):
        """
        Advance the model by one time step (one simulated day).

        Iterates over all agents sequentially, invoking each agent's
        ``step()`` method, then triggers data collection.
        """
        self.day += 1

        # Activate all agents (sequential random order via list iteration)
        for agent in self.all_agents:
            agent.step()

        # Record population-level statistics for this day
        self.datacollector.collect(self)

    def get_daily_statistics(self):
        """
        Compute comprehensive daily statistics for the current model state.

        Returns:
            dict: A dictionary containing counts for each SEIRD compartment
                  (dogs and humans separately and combined), cumulative
                  infection counts, derived metrics (attack rate, daily new
                  infections), and total deaths.
        """
        dogs = [a for a in self.all_agents if isinstance(a, DogAgent)]
        humans = [a for a in self.all_agents if isinstance(a, HumanAgent)]

        # Count agents in each health state
        dog_states = {"S": 0, "E": 0, "I": 0, "R": 0, "D": 0}
        human_states = {"S": 0, "E": 0, "I": 0, "R": 0, "D": 0}

        for dog in dogs:
            dog_states[dog.health_state] += 1
        for human in humans:
            human_states[human.health_state] += 1

        # Derived epidemiological metrics
        total_population = self.num_dogs + self.num_humans
        total_ever_infected = self.dogs_ever_infected + self.humans_ever_infected
        total_deaths = dog_states["D"] + human_states["D"]
        daily_new_infections = dog_states["E"] + human_states["E"]
        attack_rate = (
            total_ever_infected / total_population if total_population > 0 else 0
        )

        return {
            # Dog statistics
            "dogs_susceptible": dog_states["S"],
            "dogs_exposed": dog_states["E"],
            "dogs_infectious": dog_states["I"],
            "dogs_recovered": dog_states["R"],
            "dogs_dead": dog_states["D"],
            "dogs_ever_infected": self.dogs_ever_infected,
            # Human statistics
            "humans_susceptible": human_states["S"],
            "humans_exposed": human_states["E"],
            "humans_infectious": human_states["I"],
            "humans_recovered": human_states["R"],
            "humans_dead": human_states["D"],
            "humans_ever_infected": self.humans_ever_infected,
            # Combined statistics
            "total_susceptible": dog_states["S"] + human_states["S"],
            "total_exposed": dog_states["E"] + human_states["E"],
            "total_infectious": dog_states["I"] + human_states["I"],
            "total_recovered": dog_states["R"] + human_states["R"],
            "total_dead": total_deaths,
            "total_ever_infected": total_ever_infected,
            # Derived metrics
            "daily_new_infections": daily_new_infections,
            "attack_rate": attack_rate,
            "total_deaths": total_deaths,
        }

    def get_nearby_agents(self, agent, radius, agent_type=None):
        """
        Find all agents within a given radius of the specified agent.

        For large populations (> 10,000 agents), this method uses spatial
        sampling — checking a random subset of 1,000 agents rather than
        the full population — to maintain O(1)-ish performance per query.

        Args:
            agent:      The focal agent to search around.
            radius:     The search radius (Euclidean distance).
            agent_type: Optional class filter (e.g., DogAgent or HumanAgent).

        Returns:
            list: Agents within ``radius`` of ``agent``, optionally filtered.
        """
        nearby = []
        agent_x, agent_y = agent.pos

        # Performance optimisation: spatial sampling for large populations
        if len(self.all_agents) > 10000:
            sample_size = min(1000, len(self.all_agents))
            agents_to_check = self.random.sample(self.all_agents, sample_size)
        else:
            agents_to_check = self.all_agents

        for other_agent in agents_to_check:
            if other_agent == agent:
                continue

            # Euclidean distance check
            other_x, other_y = other_agent.pos
            distance = ((agent_x - other_x) ** 2 + (agent_y - other_y) ** 2) ** 0.5

            if distance <= radius:
                if agent_type is None or isinstance(other_agent, agent_type):
                    nearby.append(other_agent)

        # Apply type filter if specified (belt-and-suspenders with above)
        if agent_type:
            nearby = [a for a in nearby if isinstance(a, agent_type)]

        return nearby

    def update_state_counters(self, agent, old_state, new_state):
        """
        Incrementally update efficient state counters when an agent
        transitions between SEIRD compartments.

        This avoids the O(N) cost of re-counting all agents each step.

        Args:
            agent:     The agent that changed state.
            old_state: The previous health state string.
            new_state: The new health state string.
        """
        if isinstance(agent, DogAgent):
            # Decrement the old state counter
            if old_state == "S":
                self.dogs_s -= 1
            elif old_state == "E":
                self.dogs_e -= 1
            elif old_state == "I":
                self.dogs_i -= 1
            elif old_state == "R":
                self.dogs_r -= 1
            elif old_state == "D":
                self.dogs_d -= 1
            # Increment the new state counter
            if new_state == "S":
                self.dogs_s += 1
            elif new_state == "E":
                self.dogs_e += 1
            elif new_state == "I":
                self.dogs_i += 1
            elif new_state == "R":
                self.dogs_r += 1
            elif new_state == "D":
                self.dogs_d += 1

        elif isinstance(agent, HumanAgent):
            if old_state == "S":
                self.humans_s -= 1
            elif old_state == "E":
                self.humans_e -= 1
            elif old_state == "I":
                self.humans_i -= 1
            elif old_state == "R":
                self.humans_r -= 1
            elif old_state == "D":
                self.humans_d -= 1
            if new_state == "S":
                self.humans_s += 1
            elif new_state == "E":
                self.humans_e += 1
            elif new_state == "I":
                self.humans_i += 1
            elif new_state == "R":
                self.humans_r += 1
            elif new_state == "D":
                self.humans_d += 1
