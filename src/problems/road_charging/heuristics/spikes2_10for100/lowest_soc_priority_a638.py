from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def lowest_soc_priority_a638(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for optimizing EV charging actions with adaptive weight adjustment using reinforcement learning techniques.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "total_chargers" (int): The maximum number of available chargers.
            - "min_SoC" (float): The safety battery SoC threshold.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_solution" (Solution): The current action trajectory (solution) for EVs.
            - "battery_soc" (list[float]): A 1D array representing the battery state of charge in percentage for each EV.
            - "time_to_next_availability" (list[int]): A 1D array indicating the lead time until the fleet becomes available.
            - "operational_status" (list[int]): A 1D array indicating the operational status of each EV, where 0 represents idle, 1 represents serving a trip, and 2 represents charging.
            - "current_step" (int): The index of the current time step.
            - "reward" (float): The reward value at the current step.
        
        (Optional) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, and it will not modify the original solution.

    Returns:
        ActionOperator to assign charging actions to EVs based on their SoC and availability with adaptive weight adjustments.
        An updated dictionary with reinforcement learning metrics.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    min_SoC = global_data["min_SoC"]
    customer_arrivals = global_data["customer_arrivals"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    operational_status = state_data["operational_status"]
    current_step = state_data["current_step"]
    
    # Initialize or retrieve adaptive weights from algorithm data
    demand_weight = algorithm_data.get('demand_weight', 0.5)
    historical_weight = algorithm_data.get('historical_weight', 0.3)
    utilization_weight = algorithm_data.get('utilization_weight', 0.2)
    
    # Calculate historical charging success rates and fleet utilization metrics
    previous_steps = max(0, current_step - 5)
    historical_charging_success = np.mean([
        algorithm_data.get('charging_success_rate', [0.5] * fleet_size)[i]
        for i in range(previous_steps, current_step)
    ])
    fleet_utilization_rate = np.mean([
        operational_status[i] != 0 for i in range(fleet_size)
    ])
    
    # Reinforcement learning update for weights based on current performance
    reward = state_data.get('reward', 0)
    learning_rate = 0.01  # Small learning rate for gradual adjustment

    demand_weight += learning_rate * reward * (current_step / len(customer_arrivals))
    historical_weight += learning_rate * reward * (historical_charging_success / fleet_utilization_rate)
    utilization_weight += learning_rate * reward * (fleet_utilization_rate / historical_charging_success)

    # Normalize weights to ensure they sum up to 1
    total_weight = demand_weight + historical_weight + utilization_weight
    demand_weight /= total_weight
    historical_weight /= total_weight
    utilization_weight /= total_weight

    # Calculate priority threshold using adaptive weights
    current_arrivals = customer_arrivals[current_step]
    average_arrivals = np.mean(customer_arrivals)

    priority_threshold = max(0.1, min(0.5, (
        demand_weight * (current_arrivals / average_arrivals) +
        historical_weight * historical_charging_success +
        utilization_weight * fleet_utilization_rate
    )))
    
    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Identify EVs that are idle or non-idle with a battery_soc below the dynamically adjusted priority_threshold
    chargeable_evs = [
        i for i in range(fleet_size)
        if battery_soc[i] < priority_threshold and time_to_next_availability[i] == 0
    ]
    
    # Sort chargeable EVs by time to next availability and SoC
    chargeable_evs.sort(key=lambda i: (time_to_next_availability[i], battery_soc[i]))

    # Assign charging actions up to the number of available chargers
    for i in chargeable_evs[:total_chargers]:
        actions[i] = 1

    # Create the ActionOperator with the generated actions
    operator = ActionOperator(actions)

    # Return operator and updated algorithm data with adaptive weights
    updated_algorithm_data = {
        'demand_weight': demand_weight,
        'historical_weight': historical_weight,
        'utilization_weight': utilization_weight,
    }

    return operator, updated_algorithm_data