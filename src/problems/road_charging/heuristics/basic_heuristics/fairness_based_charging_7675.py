from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def fairness_based_charging_7675(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Rotate charging priority among EVs to ensure fair distribution of charging opportunities and avoid bias.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time for each EV. Length is fleet_size.
            - "battery_soc" (list[float]): State of charge of each EV's battery. Length is fleet_size.
            - "reward" (float): The total reward for the current time step.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "cumulative_charging_count" (list[int]): Number of charging opportunities given to each EV up to this time step. Length is fleet_size.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator with updated actions prioritizing fairness in charging distribution.
        Updated algorithm data with the new cumulative charging counts.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    cumulative_charging_count = algorithm_data.get("cumulative_charging_count", [0] * fleet_size)

    # Determine which EVs are eligible for charging (not on a ride)
    eligible_evs = [i for i in range(fleet_size) if ride_lead_time[i] == 0]

    # Sort eligible EVs by their cumulative charging count (ascending order for fairness)
    sorted_evs = sorted(eligible_evs, key=lambda i: cumulative_charging_count[i])

    # Determine how many EVs can charge at this time step
    chargers_available = min(len(sorted_evs), total_chargers)

    # Create actions with all zeros initially (meaning all EVs are not charging)
    actions = [0] * fleet_size

    # Assign charging actions to the EVs with the least charging count
    for i in range(chargers_available):
        ev_id = sorted_evs[i]
        actions[ev_id] = 1
        cumulative_charging_count[ev_id] += 1

    # Create and return the operator with the new actions
    operator = ActionOperator(actions)
    return operator, {"cumulative_charging_count": cumulative_charging_count}