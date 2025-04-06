from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def demand_based_dispatch_39ac(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ DemandBasedDispatch Heuristic with Machine Learning Feedback Mechanism.

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
            - "reward" (float): Total reward at the current time step.
            - "return" (float): Accumulated reward from step 0 to current step.

        (Optional) kwargs: Hyper-parameters for this heuristic:
            - "base_low_soc_threshold" (float, default=0.7): Base threshold for prioritizing charging.
            - "dynamic_adjustment_factor" (float, default=0.05): Factor for adjusting the threshold dynamically.
            - "adjustment_cap" (float, default=0.02): Maximum allowable change in adjustment factor.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Updated algorithm-specific data.
    """
    
    # Hyper-parameters and their defaults
    base_low_soc_threshold = kwargs.get('base_low_soc_threshold', 0.7)
    dynamic_adjustment_factor = kwargs.get('dynamic_adjustment_factor', 0.05)
    adjustment_cap = kwargs.get('adjustment_cap', 0.02)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_reward = state_data["reward"]
    accumulated_return = state_data["return"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Calculate demand at the current step
    current_demand = customer_arrivals[current_step]

    # Predict future demand using linear regression on historical data
    X = np.arange(len(customer_arrivals)).reshape(-1, 1)
    y = np.array(customer_arrivals)
    model = LinearRegression().fit(X, y)
    predicted_demand = model.predict([[current_step + 1]])[0]

    # Determine dynamic peak demand lookahead based on prediction
    if predicted_demand > current_demand:
        peak_demand_lookahead = min(16, max(1, int(predicted_demand - current_demand)))
    else:
        peak_demand_lookahead = min(16, max(1, int(current_demand - predicted_demand)))

    # Determine dynamic low SoC threshold based on demand and performance
    historical_peak_demand = max(customer_arrivals)
    if current_demand >= historical_peak_demand - peak_demand_lookahead:
        low_soc_threshold = base_low_soc_threshold - dynamic_adjustment_factor
    else:
        low_soc_threshold = base_low_soc_threshold + dynamic_adjustment_factor

    # Fine-tune dynamic adjustment factor based on serving vs charging ratio
    serving_count = sum(1 for i in range(fleet_size) if operational_status[i] == 1)
    charging_count = sum(1 for i in range(fleet_size) if operational_status[i] == 2)
    if serving_count > charging_count:
        dynamic_adjustment_factor = max(dynamic_adjustment_factor * 0.95, adjustment_cap)
    else:
        dynamic_adjustment_factor = min(dynamic_adjustment_factor * 1.05, adjustment_cap)

    # Sort EVs by battery state of charge
    sorted_evs = sorted(range(fleet_size), key=lambda i: battery_soc[i])

    # Prioritize charging for EVs with SoC below dynamic threshold
    available_chargers = total_chargers
    for i in sorted_evs:
        if operational_status[i] == 0 and battery_soc[i] < low_soc_threshold and available_chargers > 0:
            actions[i] = 1  # Set to charge
            available_chargers -= 1

    # Ensure EVs on a ride remain available
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:
            actions[i] = 0

    # Validate that the number of charging actions does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    return ActionOperator(actions), {}