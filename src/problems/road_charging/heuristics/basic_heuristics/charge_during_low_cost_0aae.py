from src.problems.base.mdp_components import Solution, ActionOperator
from typing import List, Tuple

def charge_during_low_cost_0aae(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implement the ChargeDuringLowCost heuristic algorithm.

    Args:
        global_data (dict): The global data dict containing the global instance data. In this algorithm, the following items are necessary:
            - "charging_price" (List[float]): Charging price in USD per kWh at each time step.
            - "max_time_steps" (int): Total number of time steps in the scheduling horizon.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (List[int]): Remaining ride time for each vehicle. Zero if not on ride.
            - "battery_soc" (List[float]): State of charge (SoC) of each vehicle's battery.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): Not used in this algorithm.
        (Optional and can be omitted if no used) get_state_data_function (callable): Not used in this algorithm.
        (Optional and can be omitted if no hyper parameters data) kwargs:
            - "price_threshold" (float): Default is 0. The threshold below which charging is considered economical.

    Returns:
        ActionOperator: Operator with the scheduled charging actions based on the heuristic.
        dict: Empty dictionary, as no algorithm-specific data needs to be updated.
    """
    # Extract necessary data
    charging_prices = global_data["charging_price"]
    current_time_step = len(state_data["ride_lead_time"])  # Assuming current step length equals fleet size list length
    ride_lead_time = state_data["ride_lead_time"]

    # Hyperparameter
    price_threshold = kwargs.get("price_threshold", 0)

    # Initialize actions for each EV as idle (0)
    actions = [0] * len(ride_lead_time)

    # Determine charging actions based on current time step and price threshold
    if current_time_step < global_data["max_time_steps"] and charging_prices[current_time_step] < price_threshold:
        for i, ride_time in enumerate(ride_lead_time):
            if ride_time == 0:  # Only charge if the EV is not on a ride
                actions[i] = 1

    # Create and return the operator
    operator = ActionOperator(actions)
    return operator, {}