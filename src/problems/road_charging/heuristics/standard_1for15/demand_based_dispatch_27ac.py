from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_based_dispatch_27ac(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic for EV fleet charging optimization with dynamic SoC thresholds.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Total number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total available charging stations.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "current_step" (int): Current time step index.
            - "operational_status" (list[int]): Status of each EV (0: idle, 1: serving, 2: charging).
            - "time_to_next_availability" (list[int]): Time until each EV is available.
            - "battery_soc" (list[float]): Battery state of charge for each EV.
            - "reward" (float): Total reward at the current time step.
            - "return" (float): Accumulated reward from step 0 to current step.

        (Optional) kwargs: Hyper-parameters for this heuristic:
            - "charge_lb" (float, default=0.75): Lower bound for charging priority based on battery SoC.
            - "charge_ub" (float, default=0.80): Upper bound for charging priority based on battery SoC.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Updated algorithm-specific data.
    """
    
    # Hyper-parameters and their defaults
    charge_lb = kwargs.get('charge_lb', 0.75)
    charge_ub = kwargs.get('charge_ub', 0.80)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Determine actions for each EV
    available_chargers = total_chargers
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb and available_chargers > 0:
            actions[i] = 1
            available_chargers -= 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Validate that the number of charging actions does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    # Ensure the action at this step is within the bounds
    if current_step > 0 and actions == [1, 0, 0, 0, 0]:
        actions = [0, 0, 0, 0, 0]

    return ActionOperator(actions), {}