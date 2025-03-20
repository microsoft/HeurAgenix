from src.problems.base.mdp_components import Solution, ActionOperator
import random

def random_dc6e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Randomly select EVs to charge while respecting constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - total_chargers (int): Total number of chargers available.
            - fleet_size (int): Number of EVs in the fleet.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): The current time step in the simulation.
            - ride_lead_time (list[int]): Remaining ride lead time for each EV.
            - battery_soc (list[float]): State of charge for each EV in the fleet.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        No hyper parameters are used in this heuristic.

    Returns:
        An ActionOperator that sets the charging actions for EVs.
        An empty dictionary as no algorithm data is updated in this heuristic.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions with all zeros (no charging)
    actions = [0] * fleet_size

    # List of indices of EVs that are eligible for charging
    eligible_evs = [i for i in range(fleet_size) if ride_lead_time[i] < 2 and battery_soc[i] < 1]

    # Randomly select EVs to charge, up to the number of available chargers
    num_to_charge = min(total_chargers, len(eligible_evs))
    selected_evs = random.sample(eligible_evs, num_to_charge)

    # Set the action to charge for the selected EVs
    for i in selected_evs:
        actions[i] = 1

    # Create and return the ActionOperator with the determined actions
    operator = ActionOperator(actions)
    return operator, {}