from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balance_charging_schedule_fafe(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ The Balance Charging Schedule heuristic for the Road Charging Problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
            - "order_price" (list[float]): Payments received per time step when a vehicle is on a ride.
            - "initial_charging_cost" (float): Cost incurred for the first connection to a charger.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of current time step.
            - "battery_soc" (list[float]): State of charge of each vehicle in the fleet.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "reward" (float): Total reward for all fleets for this time step.

    Returns:
        ActionOperator: Operator to apply charging actions to EVs.
        dict: Empty dictionary as no algorithm data is updated.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    order_price = global_data["order_price"]
    initial_charging_cost = global_data["initial_charging_cost"]

    current_step = state_data["current_step"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Calculate the charging balance score for each EV
    charging_balance_scores = []
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
            charging_balance_scores.append(float('-inf'))  # Cannot charge if on ride or fully charged
        else:
            reward = order_price[current_step] * (1 - battery_soc[i])
            cost = charging_price[current_step] + initial_charging_cost
            balance_score = reward - cost
            charging_balance_scores.append(balance_score)

    # Select EVs to charge based on balance scores, ensuring constraints are respected
    actions = [0] * fleet_size
    if sum(1 for score in charging_balance_scores if score > 0) > 0:
        chargers_used = 0
        for i in np.argsort(charging_balance_scores)[::-1]:
            if chargers_used < total_chargers and charging_balance_scores[i] > 0:
                actions[i] = 1
                chargers_used += 1

    return ActionOperator(actions), {}