from src.problems.base.mdp_components import Solution, ActionOperator
import random

def random_dc6e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Randomly select an action for each EV while respecting the charging constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The number of total chargers available.
            - "fleet_size" (int): The size of the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Current ride lead time for each EV. If greater than zero, the EV is on a ride and cannot charge.
            - "charging_lead_time" (list[int]): Current charging lead time for each EV.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not require specific data from this dictionary.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.
        kwargs: No hyper parameters are required by this algorithm.

    Returns:
        An ActionOperator with randomly selected actions that satisfy the charging constraints.
        An empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    charging_lead_time = state_data["charging_lead_time"]

    # Initialize actions for all EVs
    actions = [0] * fleet_size

    # Calculate available chargers considering the current charging status
    available_chargers = total_chargers - sum(1 for lead_time in charging_lead_time if lead_time > 0)

    # Randomly assign charging actions while respecting constraints
    for i in range(fleet_size):
        if ride_lead_time[i] == 0:  # Only consider EVs not on a ride
            actions[i] = random.choice([0, 1]) if available_chargers > 0 else 0
            if actions[i] == 1:
                available_chargers -= 1

    # Create an ActionOperator with the new actions
    operator = ActionOperator(actions)

    return operator, {}