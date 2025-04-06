from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_9094(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic algorithm schedules EV charging sessions focusing on prioritizing vehicles with the lowest state of charge (SoC) during cost-effective periods, while considering current demand and operational status.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "customer_arrivals" (list[int]): A list indicating the number of customer arrivals at each time step.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV is available.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): No algorithm data is required for this implementation.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.

    Returns:
        ActionOperator instance with the updated actions for EVs based on charging prioritization.
        Updated algorithm data dictionary (empty in this case as there is no algorithm data change).
    """
    
    # Extract necessary information from global_data and state_data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    fleet_size = global_data["fleet_size"]

    # Hyper-parameters
    min_customer_arrivals = kwargs.get('min_customer_arrivals', np.mean(customer_arrivals))

    # Initialize actions for the current time step
    actions = [0] * fleet_size  # Default to all EVs staying available
    
    # Determine if current step is a low demand period
    is_low_demand = customer_arrivals[current_step] < min_customer_arrivals
    
    # Sort EVs by operational status (idle first) and then by ascending order of battery SoC
    ev_indices_by_status_and_soc = sorted(
        range(fleet_size),
        key=lambda i: (operational_status[i] != 0, battery_soc[i])
    )
    
    # Track the number of chargers in use
    chargers_in_use = 0
    
    # Iterate over EVs to decide charging actions
    for ev_index in ev_indices_by_status_and_soc:
        # Skip EVs that are currently on a ride
        if operational_status[ev_index] == 1 or time_to_next_availability[ev_index] > 0:
            continue
        
        # Check if there is a free charger
        if chargers_in_use < total_chargers:
            # Charge if it is a low demand period or the EV has a low SoC
            if is_low_demand or battery_soc[ev_index] < 0.2:
                actions[ev_index] = 1  # Set action to charge
                chargers_in_use += 1

    # Create an ActionOperator instance with the determined actions
    operator = ActionOperator(actions)
    
    # Return the operator and an empty algorithm data update
    return operator, {}