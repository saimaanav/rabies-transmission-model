"""
agents.py — Agent Definitions for the Rabies Transmission Model
================================================================

This module defines the two primary agent types in the simulation:

    - DogAgent:   The reservoir host population capable of contracting,
                  incubating, and transmitting rabies via bite events.
    - HumanAgent: The susceptible population exposed to rabies through
                  dog bites, with access to post-exposure prophylaxis (PEP).

Both agent types follow the SEIRD (Susceptible -> Exposed -> Infectious ->
Recovered -> Dead) disease progression model. Movement, transmission, and
disease state transitions are governed by stochastic processes parameterized
through the parent RabiesModel instance.

Design Pattern:
    Template Method -- Each agent's ``step()`` method defines a fixed sequence
    of phases (move -> attempt bites -> update disease state -> increment timer),
    while subclasses override the specific movement and bite behaviors.

Classes:
    DogAgent    -- Primary reservoir; random-walk movement; always fatal rabies.
    HumanAgent  -- Secondary host; home-biased limited movement; PEP can save.
"""

import numpy as np
import random
from mesa import Agent


class DogAgent(Agent):
    """
    A dog agent that can contract and transmit rabies.

    Dogs are the primary reservoir for rabies in this model. They move via
    an unrestricted random walk within their roaming radius and, when
    infectious, attempt to bite all susceptible agents within the
    bite contact radius.

    Attributes:
        unique_id (int):            Unique identifier for this agent.
        health_state (str):         Current SEIRD compartment ('S','E','I','R','D').
        vaccinated (bool):          Whether the dog has been vaccinated (immune).
        roaming_radius (float):     Maximum distance the dog can move per step.
        days_in_state (int):        Days spent in the current health state.
        incubation_period (int):    Days required in 'E' before transitioning to 'I'.
        infectious_period (int):    Days required in 'I' before transitioning to 'D'.
        pos (tuple):                Current (x, y) position in continuous space.
        ever_infected (bool):       Whether this dog has ever entered an infected state.

    State Transitions:
        S -> E:  Upon successful bite transmission from an infectious agent.
        E -> I:  After ``incubation_period`` days in the Exposed state.
        I -> D:  After ``infectious_period`` days (rabies is always fatal in dogs).
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        health_state="S",
        vaccinated=False,
        roaming_radius=5.0,
    ):
        """
        Initialize a DogAgent.

        Args:
            unique_id (int):        Unique agent identifier.
            model (RabiesModel):    Reference to the parent model instance.
            pos (tuple):            Initial (x, y) position in continuous space.
            health_state (str):     Initial SEIRD state. Defaults to 'S' (Susceptible).
            vaccinated (bool):      Vaccination status. Defaults to False.
            roaming_radius (float): Max movement distance per step. Defaults to 5.0.
        """
        super().__init__(model)

        self.unique_id = unique_id
        self.health_state = health_state  # S, E, I, R, D
        self.vaccinated = vaccinated
        self.roaming_radius = roaming_radius
        self.days_in_state = 0

        # Disease progression timers -- set when transitioning to E or I
        self.incubation_period = None
        self.infectious_period = None

        # Spatial position in continuous 2D space
        self.pos = pos

        # Cumulative infection tracking for epidemiological statistics
        self.ever_infected = False
        if health_state in ["E", "I", "R", "D"]:
            self.ever_infected = True

        # If starting in the Infectious state, sample an infectious period immediately
        if health_state == "I":
            self.infectious_period = random.randint(*model.infectious_period_range)

    def step(self):
        """
        Execute one simulation step (one day) for this dog.

        Phase order is critical for correctness:
            1. Movement    -- Relocate within roaming radius (dead agents skip).
            2. Bite        -- Attempt transmission to nearby susceptible agents.
            3. Progression -- Advance disease state timers and transition if due.
            4. Timer       -- Increment ``days_in_state`` counter.
        """
        # Phase 1: Movement (dead dogs are immobile)
        if self.health_state != "D":
            self._move()

        # Phase 2: Transmission attempts (only infectious dogs bite)
        if self.health_state == "I":
            self._attempt_bites()

        # Phase 3: Disease state progression (performed AFTER bites so that
        #          a dog transitioning to Dead this step still had a chance to bite)
        self._update_disease_state()

        # Phase 4: Increment the day counter for the current state
        self.days_in_state += 1

    def _move(self):
        """
        Move the dog via an isotropic random walk within the roaming radius.

        Movement is sampled in polar coordinates:
            - angle:    Uniform on [0, 2*pi)
            - distance: Uniform on [0, roaming_radius]

        The resulting position is clamped to the grid boundaries.
        """
        # Sample random direction and distance (polar coordinates)
        angle = random.uniform(0, 2 * np.pi)
        distance = random.uniform(0, self.roaming_radius)

        # Convert polar -> Cartesian displacement
        dx = distance * np.cos(angle)
        dy = distance * np.sin(angle)

        # Compute new position, clamping to grid bounds [0, grid_size]
        new_x = max(0, min(self.model.grid_size, self.pos[0] + dx))
        new_y = max(0, min(self.model.grid_size, self.pos[1] + dy))

        # Update position in Mesa's ContinuousSpace
        self.model.space.move_agent(self, (new_x, new_y))

    def _update_disease_state(self):
        """
        Advance the disease state machine based on elapsed time in state.

        Transitions:
            E -> I:  When ``days_in_state >= incubation_period``.
            I -> D:  When ``days_in_state >= infectious_period``.
                     (Rabies is universally fatal in unvaccinated dogs.)
        """
        if self.health_state == "E":
            # Exposed -> Infectious after the stochastic incubation period
            if self.days_in_state >= self.incubation_period:
                self.health_state = "I"
                self.days_in_state = 0
                # Sample a new infectious period duration
                self.infectious_period = random.randint(
                    *self.model.infectious_period_range
                )

        elif self.health_state == "I":
            # Infectious -> Dead after the infectious period (always fatal)
            if self.days_in_state >= self.infectious_period:
                self.health_state = "D"
                self.days_in_state = 0

    def _attempt_bites(self):
        """
        Attempt to bite and infect nearby susceptible agents.

        Queries the model for all agents within the bite contact radius,
        then for each susceptible, unvaccinated agent, performs a Bernoulli
        trial with probability ``bite_transmission_prob`` to determine
        whether transmission occurs.
        """
        # Retrieve all agents within the bite contact radius
        nearby_agents = self.model.get_nearby_agents(
            self, self.model.bite_contact_radius
        )

        for agent in nearby_agents:
            # Only susceptible agents can be newly infected
            if agent.health_state == "S":
                # Check vaccination status (dogs may be vaccinated; humans are not)
                is_vaccinated = hasattr(agent, "vaccinated") and agent.vaccinated
                if not is_vaccinated:
                    # Stochastic transmission: Bernoulli trial
                    if random.random() < self.model.bite_transmission_prob:
                        agent._become_exposed()

    def _become_exposed(self):
        """
        Transition this dog from Susceptible (S) to Exposed (E).

        Samples a stochastic incubation period and updates the model's
        cumulative infection counter if this is the dog's first infection.
        """
        if self.health_state == "S":
            self.health_state = "E"
            self.days_in_state = 0
            # Sample incubation duration from the configured range
            self.incubation_period = random.randint(*self.model.incubation_period_range)

            # Update model-level epidemic tracking
            if not self.ever_infected:
                self.ever_infected = True
                self.model.dogs_ever_infected += 1


class HumanAgent(Agent):
    """
    A human agent that can contract rabies through dog (or human) bites.

    Humans exhibit limited, home-biased mobility and can access post-exposure
    prophylaxis (PEP) treatment, which significantly improves survival odds.
    Unlike dogs, humans with rabies rarely bite others (10% probability per
    step), and when they do, transmission probability is halved.

    Attributes:
        unique_id (int):            Unique identifier for this agent.
        health_state (str):         Current SEIRD compartment ('S','E','I','R','D').
        pep_access (bool):          Whether this human has access to PEP treatment.
        mobility_level (str):       Mobility category ('low', 'medium', 'high').
        vaccinated (bool):          Always False -- humans are not pre-vaccinated.
        days_in_state (int):        Days spent in the current health state.
        incubation_period (int):    Days in 'E' before transitioning to 'I'.
        infectious_period (int):    Days in 'I' before outcome determination.
        pos (tuple):                Current (x, y) position in continuous space.
        home_location (tuple):      Original position used as a home-return anchor.
        ever_infected (bool):       Whether this human has ever been infected.

    State Transitions:
        S -> E:  Upon successful bite transmission from an infectious agent.
        E -> I:  After ``incubation_period`` days in the Exposed state.
        I -> R:  If PEP accessible AND Bernoulli(pep_survival_prob) succeeds.
        I -> D:  If PEP inaccessible OR treatment fails.
    """

    def __init__(
        self,
        unique_id,
        model,
        pos,
        health_state="S",
        pep_access=False,
        mobility_level="low",
    ):
        """
        Initialize a HumanAgent.

        Args:
            unique_id (int):          Unique agent identifier.
            model (RabiesModel):      Reference to the parent model instance.
            pos (tuple):              Initial (x, y) position in continuous space.
            health_state (str):       Initial SEIRD state. Defaults to 'S'.
            pep_access (bool):        Access to post-exposure prophylaxis.
            mobility_level (str):     Movement category. Defaults to 'low'.
        """
        super().__init__(model)

        self.unique_id = unique_id
        self.health_state = health_state  # S, E, I, R, D
        self.pep_access = pep_access
        self.mobility_level = mobility_level
        self.days_in_state = 0

        # Humans are not pre-vaccinated against rabies
        self.vaccinated = False

        # Disease progression timers
        self.incubation_period = None
        self.infectious_period = None

        # Spatial attributes: current position and home anchor
        self.pos = pos
        self.home_location = pos  # Humans tend to return to their home base

        # Cumulative infection tracking
        self.ever_infected = False
        if health_state in ["E", "I", "R", "D"]:
            self.ever_infected = True

        # If starting Infectious, sample an infectious period
        if health_state == "I":
            self.infectious_period = random.randint(*model.infectious_period_range)

    def step(self):
        """
        Execute one simulation step (one day) for this human.

        Phase order mirrors DogAgent for consistency:
            1. Movement    -- Limited, home-biased movement (dead humans skip).
            2. Bite        -- Rare human-to-human transmission attempts.
            3. Progression -- Disease state advancement and outcome determination.
            4. Timer       -- Increment ``days_in_state`` counter.
        """
        # Phase 1: Limited movement (dead humans are immobile)
        if self.health_state != "D":
            self._limited_movement()

        # Phase 2: Rare bite attempts (humans with rabies can transmit)
        if self.health_state == "I":
            self._attempt_bites()

        # Phase 3: Disease state progression
        self._update_disease_state()

        # Phase 4: Increment day counter
        self.days_in_state += 1

    def _limited_movement(self):
        """
        Execute home-biased limited movement for this human.

        Movement model uses a three-outcome decision tree:
            1. Return toward home (60% probability) -- Partial movement toward
               the ``home_location``, reflecting daily commute patterns.
            2. Random exploration (15% probability) -- Isotropic random walk
               within the mobility radius.
            3. Stay in place (25% probability) -- No movement this step.

        The mobility radius is drawn from the model's ``human_mobility_radius``
        parameter, representing the average daily travel distance.
        """
        max_distance = self.model.human_mobility_radius
        move_prob = 0.15  # 15% chance of random exploration per day
        return_prob = 0.6  # 60% chance of home-return movement

        # Decision branch 1: Move toward home
        if random.random() < return_prob:
            home_distance = np.sqrt(
                (self.pos[0] - self.home_location[0]) ** 2
                + (self.pos[1] - self.home_location[1]) ** 2
            )
            if home_distance > 0.1:  # Only move if not already near home
                # Compute unit direction vector toward home
                dx = (self.home_location[0] - self.pos[0]) / home_distance
                dy = (self.home_location[1] - self.pos[1]) / home_distance

                # Move partway toward home (50% of remaining distance, capped)
                move_distance = min(max_distance, home_distance * 0.5)
                new_x = self.pos[0] + dx * move_distance
                new_y = self.pos[1] + dy * move_distance

                # Clamp to grid boundaries
                new_x = max(0, min(self.model.grid_size, new_x))
                new_y = max(0, min(self.model.grid_size, new_y))

                self.model.space.move_agent(self, (new_x, new_y))

        # Decision branch 2: Random exploration
        elif random.random() < move_prob:
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(0, max_distance)

            dx = distance * np.cos(angle)
            dy = distance * np.sin(angle)

            new_x = max(0, min(self.model.grid_size, self.pos[0] + dx))
            new_y = max(0, min(self.model.grid_size, self.pos[1] + dy))

            self.model.space.move_agent(self, (new_x, new_y))

        # Decision branch 3 (implicit): Stay in place -- no movement this step

    def _update_disease_state(self):
        """
        Advance the disease state machine based on elapsed time and PEP access.

        Transitions:
            E -> I:  When ``days_in_state >= incubation_period``.
            I -> R:  When ``days_in_state >= infectious_period`` AND ``pep_access``
                     is True AND ``Bernoulli(pep_survival_prob)`` succeeds.
            I -> D:  Otherwise (no PEP access or treatment failure).
        """
        if self.health_state == "E":
            # Exposed -> Infectious after the stochastic incubation period
            if self.days_in_state >= self.incubation_period:
                self.health_state = "I"
                self.days_in_state = 0
                self.infectious_period = random.randint(
                    *self.model.infectious_period_range
                )

        elif self.health_state == "I":
            # Infectious -> Recovery or Death after the infectious period
            if self.days_in_state >= self.infectious_period:
                if self.pep_access and random.random() < self.model.pep_survival_prob:
                    # Successful PEP treatment -> Recovery
                    self.health_state = "R"
                else:
                    # No PEP or treatment failure -> Death
                    self.health_state = "D"
                self.days_in_state = 0

    def _attempt_bites(self):
        """
        Attempt rare human-to-human rabies transmission.

        Humans with clinical rabies can occasionally bite others, but this
        is far less common than dog bites. This method applies:
            - A 10% base probability of attempting any bite this step.
            - A 50% reduction to the standard transmission probability.
        """
        # Only 10% chance that a rabid human attempts bites on a given day
        if random.random() < 0.1:
            nearby_agents = self.model.get_nearby_agents(
                self, self.model.bite_contact_radius
            )

            for agent in nearby_agents:
                if agent.health_state == "S":
                    is_vaccinated = hasattr(agent, "vaccinated") and agent.vaccinated
                    if not is_vaccinated:
                        # Halved transmission probability for human bites
                        if random.random() < (self.model.bite_transmission_prob * 0.5):
                            agent._become_exposed()

    def _become_exposed(self):
        """
        Transition this human from Susceptible (S) to Exposed (E).

        Samples a stochastic incubation period and updates the model's
        cumulative human infection counter if this is the first infection.
        """
        if self.health_state == "S":
            self.health_state = "E"
            self.days_in_state = 0
            self.incubation_period = random.randint(*self.model.incubation_period_range)

            # Update model-level epidemic tracking
            if not self.ever_infected:
                self.ever_infected = True
                self.model.humans_ever_infected += 1
