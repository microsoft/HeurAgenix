from src.problems.base.mdp_components import ActionOperator

def charging_savings_optimization_479d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Charging Savings Optimization heuristic for road_charging.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kWh at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "battery_soc" (list[float]): State of charge of each EV's battery.
        kwargs: Hyper-parameters for the heuristic algorithm.
            - "cost_saving_threshold" (float, optional): Minimum cost saving to justify rescheduling, default is 0.1.

    Returns:
        ActionOperator: The chosen operator to adjust charging actions based on cost savings.
        dict: Empty dictionary as no algorithm_data is updated.
    """

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Set default hyper-parameters
    cost_saving_threshold = kwargs.get("cost_saving_threshold", 0.1)

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Calculate potential savings for each EV
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2:
            continue  # Vehicle is on a ride, cannot charge

        if battery_soc[i] >= 1:
            continue  # Vehicle is fully charged

        # Calculate cost for current time slot
        current_cost = charging_price[current_step]

        # Find the best time slot with the lowest cost
        best_time_slot = current_step
        best_cost_saving = 0

        for future_time in range(current_step + 1, global_data["max_time_steps"]):
            future_cost = charging_price[future_time]
            cost_saving = current_cost - future_cost

            if cost_saving > best_cost_saving and cost_saving >= cost_saving_threshold:
                best_cost_saving = cost_saving
                best_time_slot = future_time

        # Schedule charging if the best time slot is found
        if best_time_slot != current_step:
            actions[i] = 0  # Postpone charging to a future time slot with lower cost
        else:
            actions[i] = 1  # Charge now if it's the best option

    # Ensure the number of charging actions does not exceed the available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}