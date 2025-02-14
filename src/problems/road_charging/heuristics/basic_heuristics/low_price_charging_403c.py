from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def low_price_charging_403c(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Implements the LowPriceCharging heuristic algorithm.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
            - "charging_price" (list[float]): Charging price in USD/kWh for each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "battery_soc" (list[float]): State of charge (SoC) for each EV in the fleet.
            - "ride_lead_time" (list[int]): The remaining ride time for each EV.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - No specific algorithm data is needed for this heuristic.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        (Optional and can be omitted if no hyper parameters data) introduction for hyper parameters in kwargs if used.
            - "low_price_threshold" (float): Default threshold to determine low charging prices. Default is 0.2.

    Returns:
        An ActionOperator instance that modifies the Solution instance to prioritize charging during low-price periods.
        An empty dictionary as no algorithm-specific data needs updating.
    """
    # Extract necessary data from inputs
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Hyper-parameters with default values
    low_price_threshold = kwargs.get("low_price_threshold", 0.2)

    # Determine the current charging price
    current_price = charging_price[current_step]

    # Initialize actions for all EVs
    actions = [0] * len(battery_soc)

    # Check if the current price is among the lowest
    sorted_prices = sorted(charging_price)
    low_prices = sorted_prices[:int(len(sorted_prices) * low_price_threshold)]

    # Prioritize charging for vehicles that meet conditions
    if current_price in low_prices:
        available_chargers = total_chargers
        for i in range(len(actions)):
            if ride_lead_time[i] < 2 and battery_soc[i] < 1:
                if available_chargers > 0:
                    actions[i] = 1
                    available_chargers -= 1

    # Create and return the ActionOperator with the actions list
    return ActionOperator(actions), {}