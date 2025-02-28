from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def farthest_cost_saving_insertion_c119(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Farthest Cost Saving Insertion heuristic for the road_charging problem.

    This algorithm selects the unscheduled EV with the highest potential cost savings from charging during low-price periods and inserts it into the schedule at a time slot that minimizes the increase in total charging cost. It considers both the availability of chargers and the current state of charge of each EV. The heuristic continues until all charging sessions are scheduled or no further sessions can be scheduled without violating charger availability constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
            - "total_chargers" (int): Total number of chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "battery_soc" (list[float]): State of charge (SoC) of each EV's battery.
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used in this algorithm.
        **kwargs: Not used in this algorithm.

    Returns:
        ActionOperator: The operator that will apply the best charging action.
        dict: Empty dictionary as no additional algorithm data is updated by this heuristic.
    """
    fleet_size = global_data["fleet_size"]
    charging_price = global_data["charging_price"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Calculate potential cost savings for each EV
    potential_savings = []
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
            potential_savings.append(float('-inf'))  # Ignore EVs on ride or fully charged
        else:
            # Calculate cost savings assuming peak pricing is the highest price in charging_price
            peak_price = max(charging_price)
            current_price = charging_price[current_step]
            savings = peak_price - current_price
            potential_savings.append(savings)

    # Select EV with highest potential savings
    best_ev = np.argmax(potential_savings)

    # Check if the best EV can be scheduled for charging
    if potential_savings[best_ev] > 0:
        actions[best_ev] = 1  # Schedule for charging

    # Ensure the total number of chargers is not exceeded
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset actions if constraints are violated

    # Create the action operator
    action_operator = ActionOperator(actions=actions)

    return action_operator, {}