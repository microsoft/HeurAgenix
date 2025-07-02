from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_49d0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_49d0 heuristic algorithm for the Road Charging Problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, ensuring it will not modify the original solution.
        kwargs (dict): Hyper parameters for controlling the algorithm behavior, including:
            - charge_lb (float, default=0.75): The battery state of charge lower bound below which charging is prioritized.
            - charge_ub (float, default=0.80): The battery state of charge upper bound above which charging is deprioritized.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on strategic allocation.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Hyper-parameters with default values
    charge_lb = kwargs.get("charge_lb", 0.75)
    charge_ub = kwargs.get("charge_ub", 0.80)

    # Initialize actions with zeros, indicating no charging by default
    actions = [0] * fleet_size

    # Determine actions for each EV
    available_chargers = total_chargers
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            if available_chargers > 0:
                actions[i] = 1
                available_chargers -= 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset to no charging if constraints are violated

    # Return ActionOperator with determined actions and no updates to algorithm data
    return ActionOperator(actions), {}