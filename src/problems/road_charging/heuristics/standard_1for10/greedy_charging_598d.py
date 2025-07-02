from src.problems.base.mdp_components import ActionOperator
import numpy as np

def greedy_charging_598d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm with dynamic charge thresholds and prioritization logic.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
            - current_step (int): The current time step index.
        kwargs: 
            - charge_lb (float, default=0.65): Lower bound threshold for battery SoC to prioritize charging.
            - charge_ub (float, default=0.85): Upper bound threshold for battery SoC to keep EVs available.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step, or an ActionOperator with all zeros if no action is taken.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]

    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    charge_lb = kwargs.get('charge_lb', 0.65)
    charge_ub = kwargs.get('charge_ub', 0.85)

    # Adjust thresholds based on fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers
    if fleet_to_charger_ratio > 5:
        charge_lb -= 0.05
        charge_ub -= 0.05

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Ensure the number of charging actions does not exceed the total chargers
    if sum(actions) > total_chargers:
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