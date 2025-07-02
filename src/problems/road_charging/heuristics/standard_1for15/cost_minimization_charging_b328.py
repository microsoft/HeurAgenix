from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_b328(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ This heuristic algorithm schedules EV charging sessions with dynamic thresholds for battery SoC based on real-time fleet utilization and demand patterns.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "customer_arrivals" (list[int]): A list indicating the number of customer arrivals at each time step.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "current_step" (int): The index of the current time step.
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV is available.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. No specific algorithm data is required for this implementation.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.
        (Optional and can be omitted if no hyper parameters data) Hyper parameters used in this algorithm:
            - "base_charge_lb" (float): Base lower bound for charging priority, defaulting to 0.20.
            - "base_charge_ub" (float): Base upper bound for charging priority, defaulting to 0.85.

    Returns:
        An ActionOperator instance with the updated actions for EVs based on charging prioritization.
        An empty dictionary as there is no algorithm data update.
    """
    # Extract necessary information from global_data and state_data
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    fleet_size = global_data["fleet_size"]

    # Base hyper-parameters for charging thresholds
    base_charge_lb = kwargs.get('base_charge_lb', 0.20)
    base_charge_ub = kwargs.get('base_charge_ub', 0.85)

    # Dynamic adjustment of charge_lb and charge_ub based on current demand
    average_customer_arrivals = np.mean(customer_arrivals)
    peak_customer_arrivals = max(customer_arrivals)
    current_customer_arrivals = customer_arrivals[current_step]

    if current_customer_arrivals < average_customer_arrivals:
        # Low demand period: raise charge_lb to keep more EVs ready for dispatch
        charge_lb = base_charge_lb + 0.05
        charge_ub = base_charge_ub
    elif current_customer_arrivals >= peak_customer_arrivals:
        # Peak demand period: lower charge_lb to prioritize charging
        charge_lb = base_charge_lb - 0.05
        charge_ub = base_charge_ub
    else:
        # Normal demand period: use base thresholds
        charge_lb = base_charge_lb
        charge_ub = base_charge_ub

    chargers_in_use = 0
    actions = [0] * fleet_size  # Default to all EVs staying available

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        elif operational_status[i] == 2 and battery_soc[i] <= charge_lb:
            actions[i] = 1
            chargers_in_use += 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
            chargers_in_use += 1
        elif current_customer_arrivals >= peak_customer_arrivals:
            if operational_status[i] == 0 and battery_soc[i] <= charge_lb:
                actions[i] = 1  # Only charge if battery SoC is critically low
                chargers_in_use += 1
            else:
                actions[i] = 0  # Remain available for dispatch

        # Ensure the number of charging actions does not exceed available chargers
        if chargers_in_use > total_chargers:
            actions[i] = 0
            chargers_in_use -= 1

    # Create an ActionOperator instance with the determined actions
    operator = ActionOperator(actions)

    # Return the operator and an empty algorithm data update
    return operator, {}