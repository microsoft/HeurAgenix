from src.problems.base.mdp_components import Solution, ActionOperator
import random

def random_selection_c849(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Randomly selects an action for each EV without considering the current state or future implications.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory (solution) for EVs.
            - current_step (int): The index of the current time step.
            - operational_status (list[int]): A list indicating the operational status (0: idle, 1: serving, 2: charging) for each EV.
            - time_to_next_availability (list[int]): A list indicating the remaining time for each EV to be available.
        get_state_data_function (callable): The function that receives the new solution as input and returns the state dictionary for the new solution.

    Returns:
        An ActionOperator containing the randomly selected actions for the current time step.
        An empty dictionary as no algorithm-specific data is updated.
    """
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']

    # Initialize actions for the current step
    actions = [0] * fleet_size

    # Determine available chargers
    available_chargers = total_chargers

    for i in range(fleet_size):
        if time_to_next_availability[i] > 0:  # EV is busy, must remain available
            actions[i] = 0
        else:
            # Randomly decide to charge or not, ensuring we don't exceed available chargers
            if available_chargers > 0:
                action = random.choice([0, 1])
                actions[i] = action
                if action == 1:
                    available_chargers -= 1
            else:
                actions[i] = 0

    # Create a new solution including the current actions
    new_solution = state_data['current_solution'].actions[:]
    if len(new_solution) <= current_step:
        new_solution.append(actions)
    else:
        new_solution[current_step] = actions

    # Return the operator with the new actions and an empty algorithm data dictionary
    return ActionOperator(actions), {}