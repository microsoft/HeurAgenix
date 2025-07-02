from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_dde9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Schedule charging sessions for EVs based on dynamic thresholds informed by real-time data analysis of customer arrivals and order prices.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "order_price" (list[float]): A list representing the payment received per minute when a vehicle is on a ride.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV is available.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "previous_actions" (list[list[int]]): A record of previous actions taken by EVs for analysis.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        kwargs: Hyper parameters for controlling the algorithm behavior:
            - base_soc_threshold (float): Default is 0.35. The base battery SoC threshold below which charging is prioritized.
            - base_fleet_to_charger_ratio_threshold (float): Default is 4.0. The base ratio above which competitive charging is necessary.

    Returns:
        ActionOperator instance with the updated actions for EVs based on strategic charging.
        Updated algorithm data dictionary (empty in this case as there is no algorithm data change).
    """
    
    # Extract necessary information from global_data and state_data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    order_price = global_data["order_price"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Extract hyper-parameters from kwargs
    base_soc_threshold = kwargs.get("base_soc_threshold", 0.35)
    base_fleet_to_charger_ratio_threshold = kwargs.get("base_fleet_to_charger_ratio_threshold", 4.0)

    # Dynamic threshold calculation based on real-time data
    soc_threshold = base_soc_threshold * (customer_arrivals[current_step] / np.mean(customer_arrivals))
    fleet_to_charger_ratio_threshold = base_fleet_to_charger_ratio_threshold * (np.mean(order_price) / order_price[current_step])

    # Initialize actions for the current time step
    num_evs = len(operational_status)
    actions = [0] * num_evs  # Default to all EVs staying available
    
    # Calculate fleet to charger ratio
    fleet_to_charger_ratio = num_evs / total_chargers
    
    # Sort EVs by ascending order of battery SoC for prioritization
    ev_indices_by_soc = np.argsort(battery_soc)
    
    # Track the number of chargers in use
    chargers_in_use = 0

    # Iterate over EVs in order of increasing SoC to decide charging actions
    for ev_index in ev_indices_by_soc:
        # Skip EVs that are currently on a ride
        if operational_status[ev_index] == 1 or time_to_next_availability[ev_index] > 0:
            continue
        
        # Check if there is a free charger
        if chargers_in_use < total_chargers:
            # Prioritize EVs with low SoC
            if battery_soc[ev_index] < soc_threshold:
                # Check if the current charging price is less than the average order price
                if charging_price[current_step] < np.mean(order_price):
                    actions[ev_index] = 1  # Set action to charge
                    chargers_in_use += 1
                elif fleet_to_charger_ratio > fleet_to_charger_ratio_threshold and customer_arrivals[current_step] > np.mean(customer_arrivals):
                    # Prioritize charging even if not the cheapest time due to high fleet-to-charger ratio and high customer demand
                    actions[ev_index] = 1
                    chargers_in_use += 1
        else:
            break  # No more chargers available

    # If no charging occurs, select the EV with the lowest SoC to charge
    if chargers_in_use == 0:
        for ev_index in ev_indices_by_soc:
            if operational_status[ev_index] == 0 and time_to_next_availability[ev_index] == 0:
                actions[ev_index] = 1
                break

    # Create an ActionOperator instance with the determined actions
    operator = ActionOperator(actions)
    
    # Return the operator and an empty algorithm data update
    return operator, {}