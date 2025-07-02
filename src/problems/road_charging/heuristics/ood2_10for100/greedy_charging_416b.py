from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_416b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm with dynamic thresholds for EV fleet charging optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - max_time_steps (int): The maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
            - current_step (int): The current time step index.
        kwargs (dict): Additional hyper-parameters for the algorithm. In this algorithm, the following items are necessary:
            - charge_lb (float): Lower bound SoC threshold for charging. Default is 0.45.
            - charge_ub (float): Upper bound SoC threshold for charging. Default is 0.50.

    Returns:
        ActionOperator: Operator indicating the actions for each EV at the current time step.
        dict: An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    max_time_steps = global_data["max_time_steps"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Hyper-parameters
    charge_lb = kwargs.get('charge_lb', 0.45)
    charge_ub = kwargs.get('charge_ub', 0.50)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Prioritize charging for idle EVs with low battery SoC
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:
            # Ensure EVs on a ride remain available
            actions[i] = 0
        elif time_to_next_availability[i] == 0:
            if battery_soc[i] <= charge_lb:
                actions[i] = 1
            elif battery_soc[i] >= charge_ub:
                actions[i] = 0

    # Limit charging actions to the number of available chargers
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