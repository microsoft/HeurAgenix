from src.problems.base.mdp_components import Solution, ActionOperator

def demand_responsive_dispatch_21fd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Heuristic algorithm for EV fleet charging optimization, incorporating feedback mechanism and predictive modeling.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "total_chargers" (int): The maximum number of available chargers.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.
            - "charging_price" (list[float]): Charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "operational_status" (list[int]): A list indicating the operational status of each EV.
            - "battery_soc" (list[float]): A list representing the battery state of charge for each EV.
            - "time_to_next_availability" (list[int]): A list indicating the time until each EV becomes available.
            - "current_step" (int): The index of the current time step.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required:
            - "base_availability_time_for_charging" (int): Base minimum idle time required for EVs to be considered for charging, default is 20 minutes.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step.
        dict: Updated algorithm data, including feedback for future adjustments.
    """

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]
    customer_arrivals = global_data["customer_arrivals"]
    charging_price = global_data["charging_price"]
    base_availability_time_for_charging = kwargs.get("base_availability_time_for_charging", 20)

    # Calculate additional metrics
    average_soc = sum(battery_soc) / fleet_size
    serving_evs = sum(1 for status in operational_status if status == 1)

    # Feedback mechanism: adjust base_availability_time_for_charging based on past performance
    past_performance = algorithm_data.get("past_performance", [])
    if past_performance:
        recent_performance = sum(past_performance[-5:]) / min(5, len(past_performance))
        if recent_performance < average_soc:
            base_availability_time_for_charging += 5
        else:
            base_availability_time_for_charging -= 5

    # Dynamic adjustment of min_availability_time_for_charging based on real-time metrics
    min_availability_time_for_charging = base_availability_time_for_charging
    if customer_arrivals[current_step] > sum(customer_arrivals) / len(customer_arrivals):  # Peak condition
        min_availability_time_for_charging -= 5
    if charging_price[current_step] < sum(charging_price) / len(charging_price):  # Low price condition
        min_availability_time_for_charging -= 5
    if average_soc < 0.5:
        min_availability_time_for_charging -= 5
    if serving_evs < fleet_size * 0.5:
        min_availability_time_for_charging += 5

    # Initialize action list for all EVs
    actions = [0] * fleet_size

    # List of EVs eligible for charging (idle and not serving a trip)
    eligible_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] <= min_availability_time_for_charging]

    # Sort eligible EVs based on SoC in ascending order (prioritize low SoC for charging)
    eligible_evs.sort(key=lambda i: (battery_soc[i], -time_to_next_availability[i]))

    # Assign charging actions up to the number of available chargers
    for i in range(min(len(eligible_evs), total_chargers)):
        actions[eligible_evs[i]] = 1

    # Ensure no EV serving a ride is assigned a charging action
    actions = [0 if time_to_next_availability[i] > 0 else actions[i] for i in range(fleet_size)]

    # Update algorithm data with feedback
    feedback = sum(battery_soc) / fleet_size
    algorithm_data["past_performance"] = algorithm_data.get("past_performance", []) + [feedback]

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, algorithm_data