from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_1af5(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_1AF5 heuristic algorithm for EV Fleet Charging Optimization with refined volatility detection.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        charging_priority_base (float, optional): Base threshold for prioritizing charging. Default is 0.4.
        demand_influence_factor (float, optional): Base factor to adjust priority threshold based on demand. Default is 0.05.
        price_influence_factor (float, optional): Base factor to adjust priority threshold based on order price. Default is 0.05.
        trend_window_size (int, optional): Number of past time steps to consider for trend analysis. Default is 10.
        volatility_threshold (float, optional): Threshold to detect high volatility in demand and price. Default is 0.1.
        smoothing_factor (float, optional): Smoothing factor for exponential moving average. Default is 0.2.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on dynamic charger availability and battery state of charge, incorporating refined volatility detection.
        An empty dictionary as this algorithm does not update algorithm data.
    """

    # Set default hyper-parameters if not provided
    charging_priority_base = kwargs.get("charging_priority_base", 0.4)
    demand_influence_factor = kwargs.get("demand_influence_factor", 0.05)
    price_influence_factor = kwargs.get("price_influence_factor", 0.05)
    trend_window_size = kwargs.get("trend_window_size", 10)
    volatility_threshold = kwargs.get("volatility_threshold", 0.1)
    smoothing_factor = kwargs.get("smoothing_factor", 0.2)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]

    # Calculate current demand and price influence
    current_demand = customer_arrivals[current_step] if current_step < len(customer_arrivals) else 0
    current_price = order_price[current_step] if current_step < len(order_price) else 0

    # Weighted trend analysis over specified window size
    past_demand = customer_arrivals[max(0, current_step - trend_window_size):current_step]
    past_price = order_price[max(0, current_step - trend_window_size):current_step]
    weights = np.arange(1, len(past_demand) + 1) if past_demand else [1]
    demand_trend = np.average(past_demand, weights=weights) if past_demand else current_demand
    price_trend = np.average(past_price, weights=weights) if past_price else current_price

    max_demand = max(customer_arrivals) if customer_arrivals else 0
    max_price = max(order_price) if order_price else 0

    # Calculate volatility in demand and price using exponential moving average
    demand_volatility = np.std(past_demand) / np.mean(past_demand) if past_demand and np.mean(past_demand) != 0 else 0
    price_volatility = np.std(past_price) / np.mean(past_price) if past_price and np.mean(past_price) != 0 else 0
    ema_demand = np.mean(past_demand) if past_demand else current_demand
    ema_price = np.mean(past_price) if past_price else current_price

    # Apply exponential moving average
    for demand in past_demand:
        ema_demand = (smoothing_factor * demand) + ((1 - smoothing_factor) * ema_demand)

    for price in past_price:
        ema_price = (smoothing_factor * price) + ((1 - smoothing_factor) * ema_price)

    # Adjust influence factors based on smoothed volatility
    dynamic_demand_influence = demand_influence_factor * (1 + demand_volatility / volatility_threshold)
    dynamic_price_influence = price_influence_factor * (1 + price_volatility / volatility_threshold)

    # Adjust charging priority threshold based on current and trend demand and price with volatility influence
    demand_ratio = (current_demand + ema_demand) / (2 * max_demand) if max_demand > 0 else 0
    price_ratio = (current_price + ema_price) / (2 * max_price) if max_price > 0 else 0
    adjusted_priority_threshold = charging_priority_base + (dynamic_demand_influence * demand_ratio) + (dynamic_price_influence * price_ratio)

    # Initialize actions with no charging (all zeros)
    actions = [0] * fleet_size

    # Identify EVs eligible for charging
    eligible_evs = [i for i in range(fleet_size) if time_to_next_availability[i] == 0 and battery_soc[i] <= adjusted_priority_threshold]
    
    # Sort eligible EVs by battery state of charge (ascending order)
    eligible_evs.sort(key=lambda x: battery_soc[x])

    # Assign charging actions to EVs with the lowest SoC, limited by charger availability
    for i in eligible_evs[:total_chargers]:
        actions[i] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}