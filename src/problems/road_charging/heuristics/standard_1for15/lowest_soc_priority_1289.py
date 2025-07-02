from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_1289(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm prioritizing EVs with the lowest SoC for charging, considering their availability and dynamically adjusting based on demand.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "customer_arrivals" (list[int]): The number of customer arrivals at each time step.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "battery_soc" (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - "time_to_next_availability" (list[int]): A 1D array indicating the lead time until the fleet becomes available.
            - "operational_status" (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.

        kwargs (dict): Contains hyper-parameters, with default values:
            - "availability_threshold" (int, default=0): Base threshold for prioritizing EVs based on their time-to-next-availability.

    Returns:
        ActionOperator to assign charging actions to EVs based on their SoC and dynamically adjusted availability threshold.
        An empty dictionary as no algorithm data needs to be updated.
    """
    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]
    current_step = state_data["current_step"]
    customer_arrivals = global_data["customer_arrivals"]
    
    # Hyper-parameters
    base_availability_threshold = kwargs.get("availability_threshold", 0)

    # Dynamically adjust availability_threshold based on demand
    demand_factor = customer_arrivals[current_step] / max(customer_arrivals)  # Normalize current demand
    availability_threshold = int(base_availability_threshold * (1 - demand_factor))

    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Collect indices of EVs that can charge (idle and available)
    chargeable_evs = [
        i for i in range(fleet_size)
        if operational_status[i] == 0 and time_to_next_availability[i] <= availability_threshold
    ]

    # Sort chargeable EVs by their state of charge (SoC) first, then by time-to-next-availability
    chargeable_evs.sort(key=lambda i: (battery_soc[i], time_to_next_availability[i]))

    # Assign charging actions to EVs with the lowest SoC
    for i in chargeable_evs[:total_chargers]:
        actions[i] = 1

    # Ensure the sum of actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, {}