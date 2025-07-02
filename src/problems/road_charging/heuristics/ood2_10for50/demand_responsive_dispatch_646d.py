from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def demand_responsive_dispatch_646d(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV fleet charging optimization with dynamic scaling and regression-based demand forecasting.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - order_price (list[float]): Payment received per minute when a vehicle is on a ride.
            - charging_price (list[float]): Charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV.
            - battery_soc (list[float]): A list representing the battery state of charge for each EV.
            - time_to_next_availability (list[int]): A list indicating the time until each EV becomes available.
            - current_step (int): The index of the current time step.
        algorithm_data (dict): The algorithm data for tracking historical performance. In this algorithm, the following items are necessary:
            - performance_history (list[float]): List of performance metrics for past time steps.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required:
            - charge_lb (float): Lower bound for battery SoC to initiate charging, default is 0.40.
            - charge_ub (float): Upper bound for battery SoC to stop charging, default is 0.70.
            - adjustment_threshold (float): Percentage threshold for dynamic adjustments, default is 0.10.
            - scaling_factor (float): Factor to scale adjustments based on historical performance and real-time data, default is 0.05.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step, ensuring the sum of actions does not exceed available chargers.
        dict: Updated algorithm data, with any necessary adjustments or feedback for future time steps.
    """
    
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = list(state_data["operational_status"])  # Convert to list to use .count()
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]

    charge_lb = kwargs.get('charge_lb', 0.40)
    charge_ub = kwargs.get('charge_ub', 0.70)
    adjustment_threshold = kwargs.get('adjustment_threshold', 0.10)
    scaling_factor = kwargs.get('scaling_factor', 0.05)

    # Initialize performance history if not present
    performance_history = algorithm_data.get('performance_history', [])

    actions = [0] * fleet_size

    # Calculate averages for dynamic adjustment
    average_customer_arrivals = sum(global_data["customer_arrivals"]) / len(global_data["customer_arrivals"])
    average_order_price = sum(global_data["order_price"]) / len(global_data["order_price"])
    
    # Dynamic scaling factor adjustment based on fleet utilization and performance
    idle_ratio = operational_status.count(0) / fleet_size
    if len(performance_history) > 5:
        recent_performance = np.mean(performance_history[-5:])
        overall_performance = np.mean(performance_history)
        performance_ratio = recent_performance / overall_performance
        scaling_factor *= performance_ratio * idle_ratio

    # Regression-based demand forecasting
    if len(global_data["customer_arrivals"]) > 10:
        X = np.arange(len(global_data["customer_arrivals"])).reshape(-1, 1)
        y = np.array(global_data["customer_arrivals"])
        model = LinearRegression().fit(X, y)
        forecast_demand = model.predict(np.array([[current_step + 1]]))[0]
        if forecast_demand > average_customer_arrivals * (1 + adjustment_threshold):
            charge_lb -= 0.05
            charge_ub += 0.05

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the number of available chargers
    if sum(actions) > total_chargers:
        # Sort by SoC to prioritize lower SoC for charging
        sorted_evs = sorted(range(fleet_size), key=lambda x: battery_soc[x])
        for i in sorted_evs[total_chargers:]:
            actions[i] = 0

    # Update performance history with the current reward
    current_performance = state_data.get('reward', 0)
    performance_history.append(current_performance)

    action_operator = ActionOperator(actions)
    return action_operator, {'performance_history': performance_history}