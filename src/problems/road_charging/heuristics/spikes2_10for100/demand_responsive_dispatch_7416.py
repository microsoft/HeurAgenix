from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_responsive_dispatch_7416(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Heuristic algorithm for EV fleet charging optimization with dynamic SoC thresholds and feedback mechanism.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "total_chargers" (int): The maximum number of available chargers.
            - "max_time_steps" (int): The total number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV becomes available.
            - "current_step" (int): The index of the current time step.
        algorithm_data (dict): The algorithm dictionary that contains historical performance data.
        kwargs: Hyper-parameters used in this algorithm:
            - "base_charge_lb" (float): Base lower bound for battery SoC to consider charging, default is 0.45.
            - "base_charge_ub" (float): Base upper bound for battery SoC to stop considering charging, default is 0.55.

    Returns:
        ActionOperator: An operator specifying the actions for each EV at the current time step.
        dict: Updated algorithm data, if necessary.
    """

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]

    # Hyper-parameters with default values
    base_charge_lb = kwargs.get("base_charge_lb", 0.45)
    base_charge_ub = kwargs.get("base_charge_ub", 0.55)

    # Historical performance data
    past_performance = algorithm_data.get("past_performance", [])
    recent_performance = np.mean(past_performance[-5:]) if len(past_performance) >= 5 else base_charge_lb

    # Calculate average SoC and adjust charge_lb and charge_ub dynamically
    average_soc = np.mean(battery_soc) if len(battery_soc) > 0 else base_charge_lb
    charge_lb = max(base_charge_lb, average_soc - 0.1, recent_performance - 0.05)
    charge_ub = min(base_charge_ub, average_soc + 0.1, recent_performance + 0.05)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions for each EV
    eligible_evs = []
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif battery_soc[i] <= charge_lb:
            eligible_evs.append(i)

    # Sort eligible EVs based on SoC in ascending order to prioritize lower SoC
    eligible_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in eligible_evs[:total_chargers]:
        actions[i] = 1

    # Update algorithm data with current performance
    current_performance = np.mean(battery_soc)  # Example performance metric
    past_performance.append(current_performance)

    # Return the operator and updated algorithm data
    action_operator = ActionOperator(actions)
    return action_operator, {"past_performance": past_performance}