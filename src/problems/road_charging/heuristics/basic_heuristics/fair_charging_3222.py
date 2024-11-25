from src.problems.base.mdp_components import Solution, ActionOperator

def fair_charging_3222(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implement max-min fairness by ensuring that no EV is left with significantly less charge compared to others, balancing the overall fleet's charging needs.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The size of the fleet.
            - "total_chargers" (int): The number of total chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): SoC of battery for each fleet.
            - "ride_lead_time" (list[int]): Ride lead time for each fleet.
        get_state_data_function (callable): The function that receives the new solution as input and returns the state dictionary for the new solution, without modifying the original solution.

    Returns:
        An ActionOperator that balances charging actions to minimize the deviation from average SoC.
        An empty dictionary as this algorithm does not update algorithm data.
    """

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Calculate the average SoC across all EVs
    average_soc = sum(battery_soc) / fleet_size

    # Initialize actions with all zeros, respecting the current solution's structure
    actions = [0] * fleet_size

    # List of EVs that can be charged, excluding those on a ride
    chargeable_evs = [(i, soc) for i, soc in enumerate(battery_soc) if ride_lead_time[i] == 0]

    # Sort EVs by their SoC in ascending order to prioritize those with lower SoC
    chargeable_evs.sort(key=lambda x: x[1])

    # Assign charging actions to achieve max-min fairness
    chargers_used = 0
    for ev_index, soc in chargeable_evs:
        if chargers_used < total_chargers:
            actions[ev_index] = 1  # Charge this EV
            chargers_used += 1

    # Create a new solution with these actions
    new_solution = Solution(actions=[actions])

    # Return an ActionOperator with the new actions and an empty dict
    return ActionOperator(actions=actions), {}