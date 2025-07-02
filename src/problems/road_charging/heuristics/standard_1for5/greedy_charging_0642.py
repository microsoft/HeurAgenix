from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def greedy_charging_0642(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ The greedy_charging_0642 heuristic algorithm optimizes EV charging decisions by integrating historical demand patterns and adaptive learning.

    Args:
        global_data (dict): Contains global instance data. Necessary keys:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - max_time_steps (int): The maximum number of time steps.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): Contains solution state data. Necessary keys:
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - time_to_next_availability (list[int]): Time remaining until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge (SoC) for each EV.
        kwargs: Hyper-parameters for the algorithm:
            - base_threshold (float, optional): Base battery SoC threshold. Default is `min_SoC + 0.1`.
            - peak_lookahead (int, optional): Initial steps to look ahead for peak customer arrivals. Default is 5.
            - feedback_factor (float, optional): Factor for incorporating feedback into threshold adjustment. Default is 0.05.

    Returns:
        ActionOperator: An operator indicating the actions for each EV at the current time step.
        dict: An updated dictionary with feedback information for future use.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Dynamic threshold calculation based on average battery SoC and feedback
    min_soc = kwargs.get("min_SoC", 0.1)
    base_threshold = kwargs.get("base_threshold", min_soc + 0.1)
    feedback_factor = kwargs.get("feedback_factor", 0.05)
    avg_soc = np.mean(battery_soc) if battery_soc.size > 0 else 0
    past_performance = algorithm_data.get("past_performance", 0)

    # Adjust threshold based on fleet's average SoC and past performance feedback
    threshold = base_threshold + (avg_soc - min_soc) * 0.1 + past_performance * feedback_factor

    # Demand prediction using historical data and machine learning
    if current_step >= 5:
        historical_data = np.array(customer_arrivals[:current_step]).reshape(-1, 1)
        future_steps = np.arange(current_step, current_step + 5).reshape(-1, 1)
        model = LinearRegression().fit(historical_data, customer_arrivals[:current_step])
        predicted_demand = model.predict(future_steps).clip(min=0)
    else:
        predicted_demand = customer_arrivals[current_step:current_step + 5]

    upcoming_peak = max(predicted_demand) if len(predicted_demand) > 0 else 0

    # Initialize actions with all zeros
    actions = [0] * fleet_size

    # Determine actions based on SoC, operational status, and predicted demand
    scores = [(battery_soc[i] + time_to_next_availability[i], i) for i in range(fleet_size)]
    scores.sort()

    chargers_used = 0
    for score, i in scores:
        if operational_status[i] == 0 and battery_soc[i] < threshold:
            # Prioritize charging during low demand periods
            if upcoming_peak < 2.84 or chargers_used < total_chargers:
                actions[i] = 1
                chargers_used += 1
            else:
                break

    # Ensure the number of charging actions does not exceed total chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    # Update algorithm data with current performance feedback using a weighted average
    current_reward = state_data.get("reward", 0)
    updated_algorithm_data = {"past_performance": (past_performance * 0.9 + current_reward * 0.1)}

    # Create and return the action operator
    operator = ActionOperator(actions)
    return operator, updated_algorithm_data