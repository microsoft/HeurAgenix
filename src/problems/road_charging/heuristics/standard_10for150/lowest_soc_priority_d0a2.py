from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_d0a2(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm using a weighted scoring system to prioritize EVs for charging based on multiple factors.

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
        ActionOperator to assign charging actions to EVs based on a weighted score of various factors.
        An empty dictionary as no algorithm data needs to be updated.
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

    # Define weights for scoring factors
    soc_weight = kwargs.get("soc_weight", 0.4)
    availability_weight = kwargs.get("availability_weight", 0.3)
    completion_weight = kwargs.get("completion_weight", 0.2)
    demand_weight = kwargs.get("demand_weight", 0.1)

    # Calculate future demand forecast for next few steps
    future_demand_forecast = sum(customer_arrivals[current_step:current_step+5])

    # Calculate scores for each EV
    scores = []
    for i in range(fleet_size):
        soc_score = 1 - battery_soc[i]  # Lower SoC gives higher score
        availability_score = 1 / (time_to_next_availability[i] + 1)  # Sooner availability gives higher score
        completion_score = 1 if operational_status[i] == 1 else 0  # Completed trip gives higher score
        demand_score = future_demand_forecast / sum(customer_arrivals)  # Higher future demand gives higher score

        # Total weighted score
        total_score = (
            soc_weight * soc_score +
            availability_weight * availability_score +
            completion_weight * completion_score +
            demand_weight * demand_score
        )
        scores.append((i, total_score))

    # Sort EVs by their total weighted score in descending order (highest score first)
    scores.sort(key=lambda x: x[1], reverse=True)

    # Assign charging actions to EVs with the highest scores
    for i, _ in scores[:total_chargers]:
        actions[i] = 1

    # Ensure the sum of actions does not exceed the number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    return operator, {}