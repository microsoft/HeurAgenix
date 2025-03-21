from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_004f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Schedule charging sessions during time steps with the lowest charging prices.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV is available.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator instance with the updated actions for EVs based on charging cost minimization.
        Updated algorithm data dictionary (empty in this case as there is no algorithm data change).
    """
    
    # Extract necessary information from global_data and state_data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions for the current time step
    num_evs = len(operational_status)
    actions = [0] * num_evs  # Default to all EVs staying available
    
    # Sort EVs by ascending order of battery SoC for prioritization
    ev_indices_by_soc = np.argsort(battery_soc)
    
    # Track the number of chargers in use
    chargers_in_use = 0

    # Iterate over EVs in order of increasing SoC to decide charging actions
    for ev_index in ev_indices_by_soc:
        # Skip EVs that are currently on a ride
        if operational_status[ev_index] == 1 or time_to_next_availability[ev_index] > 0:
            continue
        
        # Check if there is a free charger and if the EV can charge
        if chargers_in_use < total_chargers:
            # Charge if it is the cheapest time to charge
            if charging_price[current_step] == min(charging_price):
                actions[ev_index] = 1  # Set action to charge
                chargers_in_use += 1
        else:
            break  # No more chargers available

    # Create an ActionOperator instance with the determined actions
    operator = ActionOperator(actions)
    
    # Return the operator and an empty algorithm data update
    return operator, {}