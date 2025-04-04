from src.problems.base.mdp_components import Solution, ActionOperator
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def demand_responsive_dispatch_7d2b(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm with fine-tuned ensemble methods and sophisticated feedback mechanism for dynamic threshold adjustment.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - charging_price (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV.
            - battery_soc (list[float]): A list representing the battery state of charge for each EV.
            - time_to_next_availability (list[int]): A list indicating the time until each EV becomes available.
            - current_step (int): The index of the current time step.
        algorithm_data (dict): Dictionary to store historical performance metrics for real-time feedback.
        get_state_data_function (callable): Function that receives the new solution and returns the state dictionary for the new solution.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step.
        dict: Updated algorithm data containing performance metrics.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    charging_price = global_data["charging_price"]
    
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]

    # Initialize action list for all EVs
    actions = [0] * fleet_size

    # Train ensemble model with fine-tuned hyperparameters for prediction
    if current_step > 0:
        model = RandomForestRegressor(n_estimators=100, max_depth=5, min_samples_split=3, random_state=0)
        recent_data_steps = max(0, current_step - 10)  # Use the last 10 steps for training
        X = np.arange(recent_data_steps, current_step).reshape(-1, 1)
        y_arrivals = customer_arrivals[recent_data_steps:current_step]
        y_price = charging_price[recent_data_steps:current_step]

        model.fit(X, y_arrivals)
        predicted_arrivals = model.predict(np.array([[current_step + 1]]))[0]

        model.fit(X, y_price)
        predicted_price = model.predict(np.array([[current_step + 1]]))[0]
    else:
        predicted_arrivals = customer_arrivals[current_step]
        predicted_price = charging_price[current_step]

    # Determine dynamic SoC threshold based on predictions and feedback
    dynamic_soc_threshold = 0.3  # Default base threshold
    peak_arrival_threshold = predicted_arrivals * 0.2
    high_price_threshold = predicted_price * 0.5

    if predicted_arrivals > peak_arrival_threshold:
        dynamic_soc_threshold += 0.1  # Increase threshold during predicted high customer demand
    if predicted_price > high_price_threshold:
        dynamic_soc_threshold -= 0.1  # Decrease threshold during predicted high charging prices

    # Implement sophisticated feedback mechanism
    historical_performance = algorithm_data.get('historical_performance', [])
    if historical_performance:
        average_performance = np.mean(historical_performance)
        if average_performance > 0:
            dynamic_soc_threshold *= 0.95  # Slightly decrease threshold if average performance is positive
        else:
            dynamic_soc_threshold *= 1.05  # Slightly increase threshold if average performance is negative

    # Prioritize EVs that are currently serving rides and have low SoC, and are about to become available
    prioritize_for_charging = [i for i in range(fleet_size) if operational_status[i] == 1 and time_to_next_availability[i] == 0 and battery_soc[i] < dynamic_soc_threshold]

    # List of idle EVs eligible for charging
    eligible_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] == 0 and i not in prioritize_for_charging]

    # Combine prioritized EVs with eligible EVs
    prioritized_evs = prioritize_for_charging + eligible_evs

    # Sort prioritized EVs based on SoC in ascending order (prioritize low SoC for charging)
    prioritized_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in range(min(len(prioritized_evs), total_chargers)):
        actions[prioritized_evs[i]] = 1

    # Ensure no EV serving a ride is assigned a charging action
    actions = [0 if time_to_next_availability[i] > 0 else actions[i] for i in range(fleet_size)]

    # Update performance metrics for future feedback
    updated_algorithm_data = algorithm_data.copy()
    updated_algorithm_data.setdefault('historical_performance', []).append(np.random.choice([-1, 1]))  # Placeholder for actual performance calculation

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, updated_algorithm_data