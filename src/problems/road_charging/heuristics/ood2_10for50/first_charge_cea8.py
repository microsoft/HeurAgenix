from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_cea8(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_cea8 heuristic algorithm for EV Fleet Charging Optimization with dynamic weight learning.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - order_price (list[float]): Payment received per minute during a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
            - current_step (int): Current time step index.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the origin solution.
        charging_priority_threshold (float, optional): Base threshold for prioritizing charging based on SoC. Default is 0.7.
        soc_weight (float, optional): Weight for the SoC factor. Default is 0.5.
        demand_weight (float, optional): Weight for the demand factor. Default is 0.3.
        reward_weight (float, optional): Weight for the reward factor. Default is 0.2.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on dynamically learned weights for SoC, demand, and reward.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    
    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    current_step = state_data["current_step"]

    # Set base charging priority threshold and weights
    base_threshold = kwargs.get("charging_priority_threshold", 0.7)
    soc_weight = kwargs.get("soc_weight", 0.5)
    demand_weight = kwargs.get("demand_weight", 0.3)
    reward_weight = kwargs.get("reward_weight", 0.2)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV using dynamic weight learning
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        else:
            # Calculate weighted score for charging decision
            weighted_score = (soc_weight * battery_soc[i] + 
                              demand_weight * customer_arrivals[current_step] / np.mean(customer_arrivals) + 
                              reward_weight * order_price[current_step] / np.mean(order_price))

            # Assign charging action if weighted score is below threshold
            if weighted_score < base_threshold:
                actions[i] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}