from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

def first_charge_1c06(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge heuristic algorithm for the Road Charging Problem, enhanced with additional features and confidence-based adjustments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): The number of customer arrivals at each time step.
            - time_resolution (int): The length of a single time step in minutes.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving a trip, 2: charging).
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - historical_thresholds (list[float]): List of past critical_soc_threshold values used.
            - historical_results (list[float]): Corresponding performance results of past thresholds.
        kwargs: Hyper-parameters for the algorithm.
            - base_critical_soc_threshold (float): The base threshold below which an EV is considered critical for charging. Defaults to 0.9.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on charger availability and prioritization.
        An updated dictionary with historical performance data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    time_resolution = global_data["time_resolution"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    operational_status = state_data["operational_status"]
    current_step = state_data["current_step"]
    base_critical_soc_threshold = kwargs.get("base_critical_soc_threshold", 0.9)
    
    # Historical data tracking for feedback mechanism
    historical_thresholds = algorithm_data.get("historical_thresholds", [])
    historical_results = algorithm_data.get("historical_results", [])
    
    # Integrate additional features for demand prediction
    hours = np.array([i * time_resolution // 60 for i in range(len(customer_arrivals))])
    X = np.column_stack((range(len(customer_arrivals)), hours))
    y = np.array(customer_arrivals)
    model = LinearRegression().fit(X, y)
    future_hours = np.array([current_step + i * time_resolution // 60 for i in range(5)])
    future_steps = np.array([current_step + i for i in range(5)])
    predicted_demand = model.predict(np.column_stack((future_steps, future_hours))).mean()
    
    average_demand = sum(customer_arrivals) / len(customer_arrivals)
    
    # Adjust threshold based on predicted demand
    critical_soc_threshold = base_critical_soc_threshold - 0.1 if predicted_demand > average_demand else base_critical_soc_threshold
    
    # Adjust threshold based on historical performance, considering variance and confidence
    if historical_results:
        avg_result = np.mean(historical_results)
        variance_result = np.var(historical_results)
        confidence = norm.cdf((historical_results[-1] - avg_result) / np.sqrt(variance_result)) if variance_result > 0 else 0.5
        if confidence > 0.7:
            critical_soc_threshold -= 0.05 * confidence  # Scale adjustment based on confidence level
    
    # Initialize actions with zeros, indicating no charging by default
    actions = [0] * fleet_size

    # Calculate the potential EVs to prioritize for charging based on `battery_soc` and `time_to_next_availability`
    prioritized_ev_indices = [i for i in range(fleet_size) if battery_soc[i] < critical_soc_threshold and time_to_next_availability[i] == 0 and operational_status[i] != 1]

    # Iterate over prioritized EVs, assign charging action if chargers are available
    available_chargers = total_chargers
    for i in prioritized_ev_indices:
        if available_chargers > 0:
            actions[i] = 1  # Assign charging action
            available_chargers -= 1  # Reduce available chargers

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset to no charging if constraints are violated

    # Update historical data for feedback mechanism
    historical_thresholds.append(critical_soc_threshold)
    # Assuming a performance metric can be calculated here, append it to historical_results
    # historical_results.append(current_performance_metric)

    return ActionOperator(actions), {"historical_thresholds": historical_thresholds, "historical_results": historical_results}