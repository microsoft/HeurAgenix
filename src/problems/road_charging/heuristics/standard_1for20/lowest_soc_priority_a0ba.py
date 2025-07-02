from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_a0ba(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for prioritizing EVs with the lowest state of charge (SoC) for charging.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "min_SoC" (float): The safety battery SoC threshold.
            - "max_time_steps" (int): The maximum number of time steps.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "battery_soc" (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - "time_to_next_availability" (list[int]): A 1D array indicating the lead time until the fleet becomes available.
            - "operational_status" (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.

    Returns:
        ActionOperator to assign charging actions to EVs based on their SoC and availability.
        An empty dictionary as no algorithm data needs to be updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    min_SoC = global_data["min_SoC"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]

    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Prioritize idle EVs that are available and have battery_soc closest to min_SoC
    chargeable_evs = [
        i for i in range(fleet_size)
        if operational_status[i] == 0 and time_to_next_availability[i] == 0
    ]

    if not chargeable_evs:
        # If no EVs are available to charge, return zero actions
        return ActionOperator(actions), {}

    # Sort chargeable EVs by their state of charge (SoC) in ascending order,
    # prioritizing those closest to min_SoC
    chargeable_evs.sort(key=lambda i: abs(battery_soc[i] - min_SoC))

    # Assign charging actions up to the number of available chargers
    for i in chargeable_evs[:total_chargers]:
        actions[i] = 1

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, {}