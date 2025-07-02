from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np
from sklearn.linear_model import LinearRegression

def lowest_soc_priority_6658(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm using predictive modeling and real-time feedback to prioritize EVs for charging.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - total_chargers (int): The maximum number of available chargers.
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - customer_arrivals (list[int]): Projected customer arrivals for future steps.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory (solution) for EVs.
            - battery_soc (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - time_to_next_availability (list[int]): A 1D array indicating the lead time until the fleet becomes available.
            - operational_status (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.

    Returns:
        ActionOperator to assign charging actions to EVs based on predictive modeling and real-time feedback.
        Updated algorithm data containing model parameters and performance metrics.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]
    customer_arrivals = global_data["customer_arrivals"]
    current_step = state_data["current_step"]

    # Initialize actions with zeros
    actions = [0] * fleet_size

    # Predict future demand using a simple linear regression model
    if 'demand_model' not in algorithm_data:
        X = np.arange(len(customer_arrivals)).reshape(-1, 1)
        y = np.array(customer_arrivals)
        model = LinearRegression().fit(X, y)
        algorithm_data['demand_model'] = model
    else:
        model = algorithm_data['demand_model']
    
    # Predict demand for next few steps
    future_steps = np.arange(current_step, current_step + 5).reshape(-1, 1)
    future_demand_forecast = model.predict(future_steps).sum()

    # Calculate scores for each EV
    scores = []
    for i in range(fleet_size):
        soc_score = 1 - battery_soc[i]  # Lower SoC gives higher score
        availability_score = 1 / (time_to_next_availability[i] + 1)  # Sooner availability gives higher score
        completion_score = 1 if operational_status[i] == 1 else 0  # Completed trip gives higher score
        demand_score = future_demand_forecast / sum(customer_arrivals)  # Higher future demand gives higher score

        # Total weighted score
        total_score = soc_score + availability_score + completion_score + demand_score
        scores.append((i, total_score))

    # Sort EVs by their total weighted score in descending order (highest score first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Assign charging actions to EVs with the highest scores
    for i, _ in scores[:total_chargers]:
        actions[i] = 1

    # Update algorithm data with performance metrics
    previous_reward = algorithm_data.get("previous_reward", 0)
    current_reward = state_data["reward"]
    updated_algorithm_data = {
        "demand_model": model,
        "previous_reward": current_reward,
        "reward_diff": current_reward - previous_reward
    }

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, updated_algorithm_data