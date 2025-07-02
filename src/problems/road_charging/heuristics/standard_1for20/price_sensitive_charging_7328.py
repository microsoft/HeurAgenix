from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_7328(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to prioritize EV charging based on historical charging actions and battery SoC.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The total number of EVs in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "time_to_next_availability" (np.ndarray): Lead time until the fleet becomes available.
            - "battery_soc" (np.ndarray): Battery state of charge in percentage.
        kwargs: Hyper-parameters for the algorithm:
            - "charging_priority" (str, default="historical"): Strategy for prioritizing which EVs to charge.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    fleet_size = global_data["fleet_size"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_solution = state_data["current_solution"]

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Check if battery_soc data is available and not empty
    if battery_soc.size == 0 or time_to_next_availability.size == 0:
        return ActionOperator(actions), {}

    # Prioritize agents with the least charging actions historically
    if kwargs.get("charging_priority", "historical") == "historical":
        historical_charges = [sum(agent_actions) for agent_actions in current_solution.actions]
        priority_indices = np.argsort(historical_charges)
    else:
        priority_indices = np.argsort(battery_soc)

    # Add logic to check battery_soc against a critical threshold
    critical_soc_threshold = kwargs.get("critical_soc_threshold", 0.1)
    if np.all(battery_soc <= critical_soc_threshold):
        priority_indices = np.argsort([sum(agent_actions) for agent_actions in current_solution.actions])

    # Implement an if-check to evaluate the 'fleet_to_charger_ratio'
    fleet_to_charger_ratio = fleet_size / total_chargers
    if fleet_to_charger_ratio > 10:  # Example threshold, can be adjusted
        priority_indices = np.argsort(battery_soc)

    # Assign charging actions, considering constraints and priorities
    chargers_used = 0
    for i in priority_indices:
        if time_to_next_availability[i] > 0:
            actions[i] = 0  # Cannot charge if on a ride
        elif chargers_used < total_chargers:
            actions[i] = 1
            chargers_used += 1
        else:
            break

    # Ensure that the sum of actions does not exceed the number of available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}