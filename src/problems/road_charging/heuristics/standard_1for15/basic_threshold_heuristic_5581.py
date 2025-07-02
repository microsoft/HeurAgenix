from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_5581(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Adaptive Threshold Heuristic for EV Fleet Charging Optimization with Feedback Learning.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
            - charging_price (list[float]): List of charging prices at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
            - reward (float): Reward collected in the current time step.
            - return (float): Accumulated reward up to the current time step.
        algorithm_data (dict): Contains historical performance data that can be used for feedback learning.
        get_state_data_function (callable): Can be omitted if no function interface is used.
        introduction for hyper parameters in kwargs if used.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data incorporating feedback learning.
    """

    # Default and dynamic value determination
    base_charge_lb = kwargs.get('base_charge_lb', 0.74)
    base_charge_ub = kwargs.get('base_charge_ub', 0.79)
    demand_window = kwargs.get('demand_window', 5)
    
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    reward = state_data['reward']
    return_value = state_data['return']

    # Calculate fleet to charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Predict future demand
    future_demand = np.mean(customer_arrivals[current_step:current_step + demand_window]) if current_step + demand_window < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])

    # Feedback learning mechanism
    if 'historical_rewards' not in algorithm_data:
        algorithm_data['historical_rewards'] = []

    algorithm_data['historical_rewards'].append(reward)

    # Adjust thresholds based on feedback
    if len(algorithm_data['historical_rewards']) > 1:
        average_reward = np.mean(algorithm_data['historical_rewards'])
        if reward < average_reward:
            base_charge_lb -= 0.01  # Encourage more charging when performance is below average
            base_charge_ub -= 0.01  # Encourage more charging when performance is below average
        elif reward > average_reward:
            base_charge_lb += 0.01  # Restrict charging when performance is above average
            base_charge_ub += 0.01  # Restrict charging when performance is above average

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Prioritize charging for the EV with the lowest SoC at step 0
    if current_step == 0:
        sorted_indices = sorted(range(fleet_size), key=lambda i: battery_soc[i])
        actions[sorted_indices[0]] = 1

    # Ensure actions comply with constraints
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        elif battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif battery_soc[i] >= base_charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Return the operator and updated algorithm data
    return ActionOperator(actions), algorithm_data