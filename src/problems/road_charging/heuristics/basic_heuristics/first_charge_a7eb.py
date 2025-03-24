from src.problems.base.mdp_components import Solution, ActionOperator

def first_charge_a7eb(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge heuristic algorithm for the Road Charging Problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "total_chargers" (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "time_to_next_availability" (list[int]): Lead time until each EV becomes available.
            - "battery_soc" (list[float]): Battery state of charge for each EV in percentage.
            - "operational_status" (list[int]): Operational status of each EV (0: idle, 1: serving a trip, 2: charging).
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, ensuring it will not modify the original solution.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on charger availability.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]

    # Initialize actions with zeros, indicating no charging by default
    actions = [0] * fleet_size

    # Count available chargers
    available_chargers = total_chargers

    # Iterate over each EV to determine charging decisions
    for i in range(fleet_size):
        if operational_status[i] == 0 and time_to_next_availability[i] == 0:  # EV is idle and available
            if available_chargers > 0:  # If chargers are available
                actions[i] = 1  # Assign charging action
                available_chargers -= 1  # Reduce available chargers

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset to no charging if constraints are violated

    # Return ActionOperator with determined actions and no updates to algorithm data
    return ActionOperator(actions), {}