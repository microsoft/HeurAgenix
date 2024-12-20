from src.problems.base.mdp_components import *
import numpy as np

def variable_neighborhood_search_rc_116a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Variable Neighborhood Search heuristic for optimizing EV charging schedules in road charging problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "consume_rate" (list[float]): Battery consumption rate per time step for each vehicle.
            - "charging_rate" (list[float]): Battery charging rate per time step for each vehicle.
            - "charging_price" (list[float]): Charging price per kWh at each time step.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "battery_soc" (list[float]): State of charge of battery for each vehicle.

        kwargs: Hyperparameters for the search, such as neighborhood size.
            - neighborhood_size (int): The size of the neighborhood to explore. Default is 5.

    Returns:
        ActionOperator: The operator that modifies the current solution.
        dict: Empty dictionary as this function does not update algorithm_data.
    """
    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    charging_rate = global_data["charging_rate"]

    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Set default hyperparameters
    neighborhood_size = kwargs.get('neighborhood_size', 5)

    # Initialize actions for all EVs to 0
    actions = [0] * fleet_size

    # Evaluate potential charging actions within the neighborhood
    for ev in range(fleet_size):
        if ride_lead_time[ev] >= 2 or battery_soc[ev] >= 1:
            continue  # Skip vehicles on ride or fully charged

        # Check if charging is beneficial based on current price
        if current_step < len(charging_price) and charging_price[current_step] < 0:
            actions[ev] = 1  # Schedule charging for this EV

    # Ensure charger availability constraint
    if sum(actions) > total_chargers:
        for ev in range(fleet_size):
            if actions[ev] == 1:
                actions[ev] = 0  # Unassign charging if over limit
                if sum(actions) <= total_chargers:
                    break

    # Return the action operator with the new actions
    return ActionOperator(actions), {}