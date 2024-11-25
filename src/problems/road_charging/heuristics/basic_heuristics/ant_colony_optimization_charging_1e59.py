from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def ant_colony_optimization_charging_1e59(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Ant Colony Optimization for the Road Charging Problem. This function simulates the behavior of ants to optimize the scheduling of EV charging sessions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of EVs in the fleet.
            - total_chargers (int): The total number of available chargers.
            - max_time_steps (int): The maximum number of time steps in the scheduling horizon.
            - charging_price (list[float]): The charging cost per kWh at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - actions (list[list[int]]): The current charging actions for each EV at each time step.
            - battery_soc (list[int]): The current state of charge for each EV.
            - ride_lead_time (list[int]): The remaining ride time if the EV is on a ride.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - pheromone_levels (numpy.ndarray): A 2D array representing the pheromone levels on the edges between EVs and time steps.
            - desirability (numpy.ndarray): A 2D array representing the desirability of scheduling EVs to charge at specific time steps.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution.

    Hyperparameters:
        alpha (float): Pheromone importance factor. Default is 1.0.
        beta (float): Desirability importance factor. Default is 2.0.
        evaporation_rate (float): Rate at which pheromone evaporates. Default is 0.5.
        pheromone_deposit (float): Amount of pheromone deposited after a successful charging decision. Default is 1.0.

    Returns:
        ActionOperator: An operator containing the updated charging actions for the current time step.
        dict: Updated algorithm data with the new pheromone levels.
    """
    # Hyperparameters with default values
    alpha = kwargs.get('alpha', 1.0)
    beta = kwargs.get('beta', 2.0)
    evaporation_rate = kwargs.get('evaporation_rate', 0.5)
    pheromone_deposit = kwargs.get('pheromone_deposit', 1.0)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']

    current_actions = state_data['actions']
    battery_soc = state_data['battery_soc']
    ride_lead_time = state_data['ride_lead_time']

    pheromone_levels = algorithm_data.get('pheromone_levels', np.ones((fleet_size, global_data['max_time_steps'])))
    desirability = algorithm_data.get('desirability', np.ones((fleet_size, global_data['max_time_steps'])))

    # Determine the current time step
    current_time_step = len(current_actions[0])

    # If we're at the end of the time horizon or all EVs are on rides, return an empty action operator
    if current_time_step >= global_data['max_time_steps'] or all(ride_lead_time):
        return ActionOperator([[0] * fleet_size]), {}

    # Calculate probabilities for each EV to charge at the current time step
    probabilities = np.zeros(fleet_size)
    for i in range(fleet_size):
        if ride_lead_time[i] == 0:  # Only consider EVs that are not on a ride
            probabilities[i] = (pheromone_levels[i][current_time_step] ** alpha) * (desirability[i][current_time_step] ** beta)

    # Normalize probabilities
    probabilities_sum = np.sum(probabilities)
    if probabilities_sum > 0:
        probabilities /= probabilities_sum

    # Select EVs to charge based on the calculated probabilities
    charging_decisions = np.random.choice([0, 1], size=fleet_size, p=[1 - probabilities, probabilities])
    charging_decisions = np.clip(charging_decisions, 0, total_chargers - np.sum(current_actions[current_time_step]))

    # Create an ActionOperator with the new charging actions
    new_actions = current_actions[:]
    new_actions[current_time_step] = charging_decisions.tolist()

    # Update pheromone levels
    for i in range(fleet_size):
        if charging_decisions[i] == 1:
            pheromone_levels[i][current_time_step] *= (1 - evaporation_rate)
            pheromone_levels[i][current_time_step] += pheromone_deposit

    # Return the action operator and updated pheromone levels
    return ActionOperator(new_actions), {'pheromone_levels': pheromone_levels}