from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def optimal_charge_fit_13f6(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """OptimalChargeFit heuristic algorithm for the EV Fleet Charging Optimization problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The total number of EVs in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "operational_status" (list[int]): List indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - "battery_soc" (list[float]): List of current state of charge (SoC) for each EV in percentage.
            - "time_to_next_availability" (list[int]): List of time remaining for each EV to become available.
        algorithm_data (dict): (if any, can be omitted)
        get_state_data_function (callable): (if any, can be omitted)
        kwargs: Hyper-parameters used to control the algorithm behavior (if any, can be omitted).

    Returns:
        ActionOperator: The operator that assigns EVs to charging stations based on their current SoC and availability.
        dict: An empty dictionary as no algorithm-specific data updates are needed.
    """
    
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Prioritize EVs that will soon be idle and have low SoC
    priority_ev_indices = [i for i in range(fleet_size) if time_to_next_availability[i] == 0 and operational_status[i] == 1 and battery_soc[i] < 0.4]
    
    chargers_used = 0
    
    for i in priority_ev_indices:
        if chargers_used < total_chargers:
            actions[i] = 1  # Assign to charge
            chargers_used += 1
    
    # Sort remaining EVs based on their SoC (lower SoC prioritized for charging)
    remaining_ev_indices = sorted([i for i in range(fleet_size) if i not in priority_ev_indices and operational_status[i] == 0 and time_to_next_availability[i] == 0], key=lambda i: battery_soc[i])
    
    for i in remaining_ev_indices:
        if chargers_used < total_chargers:
            actions[i] = 1  # Assign to charge
            chargers_used += 1
    
    # Ensure the number of charging actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size
    
    # Generate and return the ActionOperator
    operator = ActionOperator(actions)
    
    return operator, {}