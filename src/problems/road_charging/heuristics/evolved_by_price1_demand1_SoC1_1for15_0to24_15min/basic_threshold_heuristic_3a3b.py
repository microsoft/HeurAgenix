from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

def basic_threshold_heuristic_3a3b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Enhanced Time Series Heuristic for EV Fleet Charging Optimization with External Factors.

    This heuristic uses a time series forecasting model integrated with external variables to predict future demand,
    dynamically adjusting charge thresholds to meet anticipated demand more effectively.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - charging_price (list[float]): Charging price at each time step.
            - order_price (list[float]): Order price at each time step.
            - weather_factors (list[float]): Weather impact factors at each time step (0-1 scale).
            - holiday_indicators (list[int]): Holiday indicators at each time step (0 or 1).
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. No specific items are necessary.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the origin solution.
        (Optional and can be omitted if no hyper parameters data) Hyper parameters include:
            - base_charge_lb (float): Base lower bound threshold for battery state of charge, default is 0.73.
            - base_charge_ub (float): Base upper bound threshold for battery state of charge, default is 0.78.

    Returns:
        ActionOperator: An operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Empty dictionary as no algorithm data is updated in this heuristic.
    """
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    customer_arrivals = global_data['customer_arrivals']
    weather_factors = global_data.get('weather_factors', [0] * len(customer_arrivals))
    holiday_indicators = global_data.get('holiday_indicators', [0] * len(customer_arrivals))
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']

    # Hyper-parameters with default values
    base_charge_lb = kwargs.get('base_charge_lb', 0.73)
    base_charge_ub = kwargs.get('base_charge_ub', 0.78)

    # Combine external factors with customer arrivals for the ARIMA model
    combined_data = np.array(customer_arrivals) + np.array(weather_factors) + np.array(holiday_indicators)

    # Fit an ARIMA model to predict future demand
    model = ARIMA(combined_data, order=(1, 1, 1))
    model_fit = model.fit()

    # Predict future demand for the next time step
    future_demand = model_fit.forecast(steps=1)[0]

    # Calculate average SoC
    avg_battery_soc = np.mean(battery_soc) if len(battery_soc) > 0 else 0

    # Dynamically adjust charge thresholds based on predicted demand and real-time data
    charge_lb = base_charge_lb - (future_demand / 100) + (avg_battery_soc / 10)
    charge_ub = base_charge_ub + (future_demand / 100) - (avg_battery_soc / 10)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Apply heuristic logic only if predicted future demand meets the threshold
    if future_demand >= 8:
        # Determine actions for each EV
        for i in range(fleet_size):
            # Ensure EVs on a ride remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # Prioritize charging for idle EVs with low battery SoC
            elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
                actions[i] = 1
            elif time_to_next_availability[i] == 0 and charge_lb < battery_soc[i] < charge_ub:
                actions[i] = 1 if sum(actions) < total_chargers else 0
            elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
                actions[i] = 0
    
    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        # Randomly set excess charging actions to 0 to comply with charger constraints
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}