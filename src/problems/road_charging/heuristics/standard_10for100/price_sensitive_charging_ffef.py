from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_ffef(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to optimize EV charging schedules by prioritizing low SoC vehicles, with simplified feedback and refined predictive adjustments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory for EVs.
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
            - "current_step" (int): The index of the current time step.
            - "reward" (float): The current reward for feedback mechanism.
        algorithm_data (dict): Contains historical performance data for feedback adjustment.
            - "previous_rewards" (list[float]): A list of rewards from previous time steps for feedback mechanism.
        kwargs: Hyper-parameters for better control, e.g.,:
            - "dynamic_threshold_factor" (float): Default 0.1, factor for dynamic adjustment based on real-time data trends.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Updated algorithm data with current rewards added to previous rewards.
    """
    # Extract necessary data
    total_chargers = global_data.get("total_chargers")
    charging_price = global_data.get("charging_price")
    customer_arrivals = global_data.get("customer_arrivals")
    current_step = state_data.get("current_step")
    time_to_next_availability = state_data.get("time_to_next_availability")
    battery_soc = state_data.get("battery_soc")
    previous_rewards = algorithm_data.get("previous_rewards", [])
    current_reward = state_data.get("reward", 0)

    # Dynamic threshold adjustment based on real-time data, simplified feedback, and refined predictive component
    dynamic_threshold_factor = kwargs.get("dynamic_threshold_factor", 0.1)
    recent_prices = charging_price[max(0, current_step-10):current_step]
    avg_recent_price = np.mean(recent_prices) if recent_prices else 0.30
    avg_charging_price_threshold = avg_recent_price * (1 + dynamic_threshold_factor)

    # Simplified feedback mechanism to adjust charging price threshold
    if previous_rewards:
        avg_previous_reward = np.mean(previous_rewards)
        if avg_previous_reward < current_reward:  # Current reward is higher
            avg_charging_price_threshold *= (1 + dynamic_threshold_factor)
        else:  # Current reward is lower
            avg_charging_price_threshold *= (1 - dynamic_threshold_factor)

    # Predictive adjustment for demand and price spikes using larger time windows
    predicted_demand_spike = np.mean(customer_arrivals[current_step:current_step+10]) > kwargs.get("average_customer_arrivals", 2.84)
    predicted_price_spike = np.mean(charging_price[current_step:current_step+10]) > avg_recent_price

    if predicted_demand_spike or predicted_price_spike:
        avg_charging_price_threshold *= (1 + dynamic_threshold_factor)

    fleet_size = len(time_to_next_availability)
    actions = [0] * fleet_size
    
    if not charging_price or not customer_arrivals:
        return ActionOperator(actions), {}

    # Calculate future demand and prioritize charging during low-demand periods
    future_demand = np.mean(customer_arrivals[current_step:])
    avg_customer_arrivals = kwargs.get("average_customer_arrivals", 2.84)
    low_demand = future_demand < avg_customer_arrivals

    # Sort EVs by battery SoC in ascending order (prioritizing lower SoC for charging)
    ev_indices = np.argsort(battery_soc)

    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            actions[i] = 0
        elif chargers_used < total_chargers and (charging_price[current_step] <= avg_charging_price_threshold or low_demand):
            actions[i] = 1
            chargers_used += 1
        else:
            break

    # Update algorithm data with current rewards
    updated_algorithm_data = {"previous_rewards": previous_rewards + [current_reward]}

    return ActionOperator(actions), updated_algorithm_data