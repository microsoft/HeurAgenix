from src.problems.base.mdp_components import Solution, ActionOperator
from sklearn.linear_model import LinearRegression
import numpy as np

def demand_based_dispatch_6dd9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """DemandBasedDispatch 6DD9 Heuristic: Allocates EVs to remain available or go to charge based on machine learning demand predictions.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Total number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total available charging stations.
            - "customer_arrivals" (list[int]): Number of customer arrivals at each time step.

        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "current_step" (int): Current time step index.
            - "operational_status" (list[int]): Status of each EV (0: idle, 1: serving, 2: charging).
            - "time_to_next_availability" (list[int]): Time until each EV is available.
            - "battery_soc" (list[float]): Battery state of charge for each EV.

        kwargs: Optional hyper-parameters for this heuristic:
            - "base_low_soc_threshold" (float, default=0.3): Base threshold below which EVs are prioritized for charging.
            - "average_charging_rate" (float, default=0.02): Average charging rate, used to prevent unnecessary charging.
            - "low_demand_factor" (float, default=0.5): Factor of max demand to identify low demand periods.
            - "demand_forecast_window" (int, default=5): The number of future time steps to consider for demand forecasting.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Updated algorithm-specific data if applicable.
    """
    
    base_low_soc_threshold = kwargs.get('base_low_soc_threshold', 0.3)
    average_charging_rate = kwargs.get('average_charging_rate', 0.02)
    low_demand_factor = kwargs.get('low_demand_factor', 0.5)
    demand_forecast_window = kwargs.get('demand_forecast_window', 5)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Prepare data for machine learning model
    X = np.arange(len(customer_arrivals)).reshape(-1, 1)
    y = np.array(customer_arrivals)

    # Train a simple linear regression model for demand prediction
    model = LinearRegression()
    model.fit(X, y)

    # Forecast future demand using the model
    future_steps = np.arange(current_step, min(current_step + demand_forecast_window, len(customer_arrivals))).reshape(-1, 1)
    future_demand = np.mean(model.predict(future_steps))

    # Determine the peak customer arrivals from the data
    peak_customer_arrivals = max(customer_arrivals) if customer_arrivals else 0

    # Dynamic adjustment of low_soc_threshold based on predicted demand
    demand_ratio = future_demand / peak_customer_arrivals if peak_customer_arrivals > 0 else 0
    low_soc_threshold = base_low_soc_threshold * (1 + (1 - demand_ratio))

    # Determine the number of EVs that can charge based on charger availability
    available_chargers = total_chargers

    # Identify EVs that should prioritize charging
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:  # EV is on a ride
            actions[i] = 0  # Must remain available
        elif battery_soc[i] < low_soc_threshold and available_chargers > 0:
            actions[i] = 1  # Set to charge
            available_chargers -= 1

    # If demand is significantly lower than peak, allow charging
    if future_demand < peak_customer_arrivals * low_demand_factor:
        # Prioritize charging for EVs with battery_soc less than average charging rate
        for i in range(fleet_size):
            if battery_soc[i] > average_charging_rate and available_chargers < total_chargers:
                actions[i] = 0  # Reset to remain available
                available_chargers += 1

    # Ensure the sum of actions does not exceed total chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    return ActionOperator(actions), {}