from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def optimal_charge_fit_1636(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """OptimalChargeFit heuristic algorithm for the EV Fleet Charging Optimization problem with adaptive learning and external data integration.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "fleet_size" (int): The total number of EVs in the fleet.
            - "customer_arrivals" (list[int]): Predicted customer arrivals at each time step.
            - "external_data" (dict): External data such as weather, traffic conditions, etc., impacting demand.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): An instance of the Solution class representing the current solution.
            - "operational_status" (list[int]): List indicating the operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - "battery_soc" (list[float]): List of current state of charge (SoC) for each EV in percentage.
            - "time_to_next_availability" (list[int]): List of time remaining for each EV to become available.
            - "reward" (float): The reward accumulated from previous actions (if used for feedback).
        algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - "soc_threshold_history" (list[float]): History of SoC thresholds used in past steps.
            - "recent_rewards" (list[float]): Recent rewards to influence threshold adjustment.
        get_state_data_function (callable): (if any, can be omitted)
        kwargs: Hyper-parameters used to control the algorithm behavior (if any, can be omitted).

    Returns:
        ActionOperator: The operator that assigns EVs to charging stations based on their current SoC and availability with adaptive learning.
        dict: Updated algorithm data including the history of SoC thresholds and recent rewards.
    """
    
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]
    reward = state_data.get("reward", 0)
    soc_threshold_history = algorithm_data.get("soc_threshold_history", [])
    recent_rewards = algorithm_data.get("recent_rewards", [])
    external_data = global_data.get("external_data", {})  # Example external data usage
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Calculate dynamic SoC threshold using predictive modeling with external data
    average_soc = np.mean(battery_soc)
    predicted_demand_peak = max(customer_arrivals[current_step:current_step+5])  # Look ahead 5 steps for peak demand
    rolling_average_demand = np.mean(customer_arrivals[max(0, current_step-5):current_step])  # Rolling average of past demand
    external_factor = external_data.get("impact_factor", 1.0)  # Example: adjust demand based on external factor
    soc_threshold = max(0.2, min(0.4, average_soc * 0.75 * (predicted_demand_peak / rolling_average_demand) * external_factor))  # Initial dynamic threshold logic

    # Adaptive learning: Adjust soc_threshold based on weighted recent rewards and external data
    recent_rewards.append(reward)
    if len(recent_rewards) > 5:
        recent_rewards.pop(0)  # Keep only the last 5 rewards for weighting
    weighted_reward_influence = np.mean(recent_rewards) / 10  # Example weighting factor
    soc_threshold = max(0.2, min(0.4, soc_threshold * (1 + weighted_reward_influence)))

    # Store the threshold in history for future feedback
    soc_threshold_history.append(soc_threshold)
    
    # Prioritize EVs that will soon be idle and have low SoC
    priority_ev_indices = [i for i in range(fleet_size) if time_to_next_availability[i] == 0 and operational_status[i] == 1 and battery_soc[i] < soc_threshold]
    
    chargers_used = 0
    
    for i in priority_ev_indices:
        if chargers_used < total_chargers:
            actions[i] = 1  # Assign to charge
            chargers_used += 1
    
    # Sort remaining EVs based on their SoC (lower SoC prioritized for charging)
    remaining_ev_indices = sorted([i for i in range(fleet_size) if i not in priority_ev_indices and operational_status[i] == 0 and time_to_next_availability[i] == 0], key=lambda i: battery_soc[i])
    
    for i in remaining_ev_indices:
        if chargers_used < total_chargers:
            actions[i] = 1  # Assign to charge
            chargers_used += 1
    
    # Ensure the number of charging actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size
    
    # Generate and return the ActionOperator
    operator = ActionOperator(actions)
    
    # Return updated algorithm data with the threshold history and recent rewards
    return operator, {"soc_threshold_history": soc_threshold_history, "recent_rewards": recent_rewards}