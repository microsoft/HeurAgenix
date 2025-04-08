from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def demand_based_dispatch_6e2f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Adaptive heuristic for EV fleet charging optimization with enhanced machine learning-based demand prediction.

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
            - "base_charge_lb" (float, default=0.75): Base lower bound for charging priority based on battery SoC.
            - "base_charge_ub" (float, default=0.80): Base upper bound for charging priority based on battery SoC.
            - "demand_sensitivity" (float, default=0.05): Sensitivity factor for adjusting thresholds based on demand.
            - "sensitivity_cap" (float, default=0.02): Maximum allowable adjustment to sensitivity.
            - "demand_window" (int, default=5): Number of time steps to consider for calculating weighted average demand.

    Returns:
        ActionOperator: An operator containing the new action set for the current time step.
        dict: Updated algorithm-specific data.
    """
    
    # Hyper-parameters and their defaults
    base_charge_lb = kwargs.get('base_charge_lb', 0.75)
    base_charge_ub = kwargs.get('base_charge_ub', 0.80)
    demand_sensitivity = kwargs.get('demand_sensitivity', 0.05)
    sensitivity_cap = kwargs.get('sensitivity_cap', 0.02)
    demand_window = kwargs.get('demand_window', 5)

    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]

    actions = [0] * fleet_size  # Initialize actions to remain available

    # Prepare features for the machine learning model
    X = np.array([[i, i % 24, i % 7] for i in range(len(customer_arrivals))])
    y = np.array(customer_arrivals)

    # Use a Random Forest model to predict future demand
    model = RandomForestRegressor().fit(X, y)
    future_features = np.array([[current_step + 1, (current_step + 1) % 24, (current_step + 1) % 7]])
    predicted_demand = model.predict(future_features)[0]

    # Calculate weighted average demand over the defined window
    start_index = max(0, current_step - demand_window + 1)
    window_demands = customer_arrivals[start_index:current_step+1]
    weighted_avg_demand = np.average(window_demands, weights=np.arange(1, len(window_demands) + 1))

    # Calculate dynamic thresholds based on predicted and weighted average demand
    peak_demand = np.max(customer_arrivals)
    adjusted_sensitivity = min(max(demand_sensitivity, -sensitivity_cap), sensitivity_cap)

    charge_lb = base_charge_lb + adjusted_sensitivity * (predicted_demand - weighted_avg_demand) / peak_demand
    charge_ub = base_charge_ub + adjusted_sensitivity * (predicted_demand - weighted_avg_demand) / peak_demand

    # Determine actions for each EV
    available_chargers = total_chargers
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb and available_chargers > 0:
            actions[i] = 1
            available_chargers -= 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Validate that the number of charging actions does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Default to all zero actions if constraints are violated

    # Ensure the action at this step is within the bounds
    if current_step > 0 and actions == [1, 0, 0, 0, 0]:
        actions = [0, 0, 0, 0, 0]

    # Update historical rewards
    historical_rewards = algorithm_data.get('historical_rewards', [])
    historical_rewards.append(state_data["reward"])
    algorithm_data['historical_rewards'] = historical_rewards

    return ActionOperator(actions), algorithm_data