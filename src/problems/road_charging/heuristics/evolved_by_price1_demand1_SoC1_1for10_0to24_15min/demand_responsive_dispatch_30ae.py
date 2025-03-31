from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def demand_responsive_dispatch_30ae(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic for EV Fleet Charging Optimization using machine learning for demand prediction.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - fleet_to_charger_ratio (float): Ratio of fleet size to total chargers.
            - time_of_day_factor (list[float]): Factor representing demand variation by time of day.
            - weather_factor (float): Factor representing demand variation due to weather conditions.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): Function that receives the new solution and returns the state dictionary for the new solution.
        kwargs: Hyper-parameters for controlling algorithm behavior. Defaults are set as required.
            - base_charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.2.
            - base_charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.5.
            - feedback_interval (int, optional): Interval for feedback loop to adjust thresholds. Default is 5.
            - model_update_interval (int, optional): Interval for updating the prediction model. Default is 10.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.2)
    base_charge_ub = kwargs.get('base_charge_ub', 0.5)
    feedback_interval = kwargs.get('feedback_interval', 5)
    model_update_interval = kwargs.get('model_update_interval', 10)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    time_of_day_factor = global_data.get('time_of_day_factor', [1.0] * len(customer_arrivals))
    weather_factor = global_data.get('weather_factor', 1.0)

    # Calculate peak customer arrivals if not provided
    peak_customer_arrivals = kwargs.get('peak_customer_arrivals', max(customer_arrivals))
    fleet_to_charger_ratio = global_data['fleet_size'] / global_data['total_chargers']

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Initialize or update the demand prediction model
    if 'demand_model' not in algorithm_data or current_step % model_update_interval == 0:
        historical_data = np.array([customer_arrivals[i-feedback_interval:i] for i in range(feedback_interval, len(customer_arrivals))])
        targets = np.array(customer_arrivals[feedback_interval:])
        algorithm_data['demand_model'] = RandomForestRegressor(n_estimators=100)
        algorithm_data['demand_model'].fit(historical_data, targets)

    # Predict future demand using the model
    if current_step >= feedback_interval:
        recent_data = np.array(customer_arrivals[current_step-feedback_interval:current_step]).reshape(1, -1)
        predicted_demand = algorithm_data['demand_model'].predict(recent_data)[0]
    else:
        predicted_demand = customer_arrivals[current_step]

    # Adjust demand prediction with additional factors
    time_factor = time_of_day_factor[current_step] if current_step < len(time_of_day_factor) else 1.0
    adjusted_demand = predicted_demand * time_factor * weather_factor

    # Implement safety margin in threshold adjustments
    safety_margin = 0.05
    charge_lb = max(base_charge_lb, 0.5 - 0.1 * (fleet_to_charger_ratio / 10) * (adjusted_demand / peak_customer_arrivals) - safety_margin)
    charge_ub = min(base_charge_ub, 0.6 + 0.1 * (adjusted_demand / np.max(customer_arrivals)) + safety_margin)

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low time_to_next_availability and battery_soc
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and updated algorithm data
    return ActionOperator(actions), algorithm_data