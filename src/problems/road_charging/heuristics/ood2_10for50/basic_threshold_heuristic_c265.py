from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def basic_threshold_heuristic_c265(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Advanced Predictive Heuristic for EV Fleet Charging Optimization using Machine Learning.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - consume_rate (list[float]): Battery consumption rate per time step for each vehicle.
            - historical_data (dict): Historical data for predicting future demand patterns.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Contains data used for machine learning feedback.
        get_state_data_function (callable): Function to receive the new solution as input and return the state dictionary for new solution.
        kwargs: Hyper-parameters used in this heuristic:
            - base_charge_lb (float, default=0.3): Base lower bound of SoC for charging decisions.
            - base_charge_ub (float, default=0.4): Base upper bound of SoC for staying available.
            - dynamic_demand_factor (float, default=0.5): Factor to adjust demand threshold dynamically.
            - prediction_window (int, default=5): Number of previous time steps to consider for prediction.

    Returns:
        ActionOperator: An operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data with machine learning feedback.
    """

    # Set base values for hyper-parameters if not provided
    base_charge_lb = kwargs.get('base_charge_lb', 0.3)
    base_charge_ub = kwargs.get('base_charge_ub', 0.4)
    dynamic_demand_factor = kwargs.get('dynamic_demand_factor', 0.5)
    prediction_window = kwargs.get('prediction_window', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    customer_arrivals = global_data['customer_arrivals']
    consume_rate = global_data['consume_rate']
    historical_data = global_data.get('historical_data', {})
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']

    # Calculate dynamic thresholds based on average SoC, ensuring non-empty battery_soc
    avg_soc = np.mean(battery_soc) if len(battery_soc) > 0 else 0.0
    charge_lb = max(base_charge_lb, avg_soc - 0.05)
    charge_ub = min(base_charge_ub, avg_soc + 0.05)

    # Predict future customer demand using machine learning model
    past_arrivals = historical_data.get('customer_arrivals', customer_arrivals[max(0, current_step-prediction_window):current_step])
    if len(past_arrivals) >= prediction_window:
        X = np.arange(len(past_arrivals)).reshape(-1, 1)
        y = np.array(past_arrivals)
        model = LinearRegression().fit(X, y)
        predicted_demand = model.predict(np.array([[current_step]]))[0] * dynamic_demand_factor
    else:
        predicted_demand = np.mean(past_arrivals) * dynamic_demand_factor if past_arrivals else dynamic_demand_factor

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        else:
            # Calculate expected depletion and future needs
            expected_depletion = consume_rate[i] * (max(customer_arrivals) - current_step)
            future_soc = battery_soc[i] - expected_depletion

            # Prioritize charging based on predicted demand and future SoC
            if customer_arrivals[current_step] <= predicted_demand:
                if future_soc < charge_lb:
                    actions[i] = 1
            elif battery_soc[i] <= charge_lb:
                actions[i] = 1

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

    # Update algorithm data with feedback from machine learning model
    updated_algorithm_data = {"predicted_demand": predicted_demand}

    # Return the operator and updated algorithm data
    return ActionOperator(actions), updated_algorithm_data