from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def optimal_charge_fit_31dd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """OptimalChargeFit31dd heuristic algorithm for the EV Fleet Charging Optimization problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The total number of EVs in the fleet.
            - "average_customer_arrivals" (float): The average number of customer arrivals per time step.
            - "peak_customer_arrivals" (int): The peak number of customer arrivals in any time step.
            - "customer_arrivals" (list[int]): List of customer arrivals for each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "operational_status" (list[int]): List indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - "battery_soc" (list[float]): List of current state of charge (SoC) for each EV in percentage.
            - "time_to_next_availability" (list[int]): List of time remaining for each EV to become available.
        algorithm_data (dict): (if any, can be omitted)
        get_state_data_function (callable): (if any, can be omitted)
        kwargs: Hyper-parameters used to control the algorithm behavior (if any, can be omitted).

    Returns:
        ActionOperator: The operator that assigns EVs to charging stations based on their current SoC and predicted demand.
        dict: An empty dictionary as no algorithm-specific data updates are needed.
    """
    
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data.get("current_step", 0)
    
    # Default value for peak_customer_arrivals to avoid KeyError
    peak_customer_arrivals = global_data.get("peak_customer_arrivals", max(customer_arrivals))
    
    # Calculate future demand based on customer arrivals
    future_demand = sum(customer_arrivals[current_step:current_step+5])  # Look-ahead window of 5 time steps
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Sort EVs based on their SoC and prioritize idle EVs with lowest SoC
    ev_indices = sorted(range(fleet_size), key=lambda i: (operational_status[i] == 0, battery_soc[i]))
    
    chargers_used = 0
    
    for i in ev_indices:
        # Check if the EV is idle and can be charged
        if operational_status[i] == 0 and time_to_next_availability[i] == 0:
            if chargers_used < total_chargers:
                # Dynamic threshold based on future demand
                dynamic_threshold = (future_demand / (len(customer_arrivals) - current_step)) / peak_customer_arrivals
                if battery_soc[i] < dynamic_threshold:
                    actions[i] = 1  # Assign to charge
                    chargers_used += 1
    
    # Ensure the number of charging actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size
    
    # Ensure an action is taken if there are idle EVs and chargers available
    if chargers_used == 0:
        for i in ev_indices:
            if operational_status[i] == 0 and time_to_next_availability[i] == 0:
                actions[i] = 1
                break
    
    # Generate and return the ActionOperator
    operator = ActionOperator(actions)
    
    return operator, {}