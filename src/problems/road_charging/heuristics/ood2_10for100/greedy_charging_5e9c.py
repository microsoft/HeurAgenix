from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_5e9c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm prioritizes charging EVs with the lowest battery SoC.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - max_time_steps (int): The maximum number of time steps.
            - average_customer_arrivals (float): The average number of customer arrivals.
            - peak_customer_arrivals (int): The peak number of customer arrivals.
            - average_order_price (float): The average order price.
            - average_charging_price (float): The average charging price.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
            - current_step (int): The index of the current time step.
        kwargs: Hyper-parameters for algorithm control:
            - charge_lb (float, optional): Lower bound of SoC for charging. Default is 0.47.
            - charge_ub (float, optional): Upper bound of SoC for charging. Default is 0.55.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]
    
    # Retrieve or set default hyper-parameters
    charge_lb = kwargs.get("charge_lb", 0.47)
    charge_ub = kwargs.get("charge_ub", 0.55)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions based on battery SoC and operational status
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0
    
    # Limit the number of charging actions to the number of available chargers
    if sum(actions) > total_chargers:
        # Sort EV indices by battery SoC to prioritize those with the lowest SoC
        soc_indices = np.argsort(battery_soc)
        chargers_used = 0
        for idx in soc_indices:
            if chargers_used < total_chargers and actions[idx] == 1:
                chargers_used += 1
            else:
                actions[idx] = 0

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, {}