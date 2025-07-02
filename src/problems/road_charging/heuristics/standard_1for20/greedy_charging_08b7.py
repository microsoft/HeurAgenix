from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_08b7(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ GreedyCharging heuristic algorithm adjusts EV charging decisions using a dynamic weighting system based on historical performance data.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - order_price (list[float]): A list representing the payment received per minute when a vehicle is on a ride.
            - charging_price (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): A list indicating the time remaining until each EV becomes available.
            - battery_soc (list[float]): A list representing the battery state of charge (SoC) for each EV.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - historical_performance (list[float]): A list representing historical performance data to adjust weighting dynamically.
        get_state_data_function (callable): The function that receives the new solution as input and returns the state dictionary for the new solution, ensuring it does not modify the original solution.
        threshold_peak (float, optional): The base SoC threshold for charging during peak demand, defaulting to 0.5.
        initial_price_weight (float, optional): The initial weight given to economic factors when determining the charging threshold, defaulting to 0.5.

    Returns:
        ActionOperator: An operator that indicates the actions for each EV at the current time step.
        dict: Updated algorithm data including potential adjustments to historical performance.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]
    charging_price = global_data["charging_price"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Retrieve historical performance data
    historical_performance = algorithm_data.get("historical_performance", [0.5] * fleet_size)

    # Set dynamic threshold based on peak demand and economic factors
    threshold_peak = kwargs.get("threshold_peak", 0.5)
    initial_price_weight = kwargs.get("initial_price_weight", 0.5)

    average_customer_arrivals = np.mean(customer_arrivals)
    peak_customer_arrivals = max(customer_arrivals)

    # Calculate weighted threshold considering both demand and price factors with dynamic weighting adjustment
    demand_factor = threshold_peak if customer_arrivals[current_step] > average_customer_arrivals else kwargs.get("threshold", 0.7)
    price_weight = np.mean(historical_performance)  # Dynamic adjustment using historical data
    price_factor = (order_price[current_step] - charging_price[current_step]) * price_weight
    threshold = max(demand_factor, price_factor)

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions based on battery SoC, operational status, and weighted threshold
    for i in range(fleet_size):
        # If the EV is on a ride, it cannot charge
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # If the EV is idle and its SoC is below the weighted threshold, consider charging
        elif operational_status[i] == 0 and battery_soc[i] < threshold:
            actions[i] = 1

    # Prioritize EVs with the lowest SoC when the fleet_to_charger_ratio is high
    if sum(actions) > total_chargers:
        soc_indices = np.argsort(battery_soc)
        chargers_used = 0
        for idx in soc_indices:
            if chargers_used < total_chargers and operational_status[idx] == 0 and battery_soc[idx] < threshold:
                actions[idx] = 1
                chargers_used += 1
            else:
                actions[idx] = 0

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, {"historical_performance": historical_performance}