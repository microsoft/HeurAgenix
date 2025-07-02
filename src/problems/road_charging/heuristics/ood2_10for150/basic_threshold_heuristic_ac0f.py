from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_ac0f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Threshold Heuristic with Feedback Loop for EV Fleet Charging Optimization.

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
            - reward (float): The total reward for the entire fleet at the current time step.
            - return (float): The accumulated reward for the entire fleet from time step 0 to the current time step.
        algorithm_data (dict): Contains historical performance data for feedback loop adjustments.
        get_state_data_function (callable): Not used directly, placeholder for function interface.
        introduction for hyper parameters in kwargs if used:
            - base_charge_lb (float, optional): Base lower bound of SoC for charging decisions. Default is 0.75.
            - base_charge_ub (float, optional): Base upper bound of SoC for staying available. Default is 0.76.
            - demand_window (int, optional): Number of time steps to consider for demand prediction. Default is 5.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data with feedback metrics.
    """

    # Set default values for hyper-parameters
    base_charge_lb = kwargs.get('base_charge_lb', 0.75)
    base_charge_ub = kwargs.get('base_charge_ub', 0.76)
    demand_window = kwargs.get('demand_window', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    order_price = global_data['order_price']
    charging_price = global_data['charging_price']
    reward = state_data['reward']
    return_value = state_data['return']

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Enhanced demand prediction using rolling variance
    recent_demand = customer_arrivals[max(0, current_step - demand_window):current_step]
    future_demand = np.mean(recent_demand) if len(recent_demand) > 0 else np.mean(customer_arrivals[:current_step + demand_window])
    demand_variance = np.var(recent_demand) if len(recent_demand) > 0 else np.var(customer_arrivals[:current_step + demand_window])

    # Feedback loop: evaluate threshold adjustments against performance metrics
    if 'previous_reward' in algorithm_data:
        performance_change = reward - algorithm_data['previous_reward']
        if performance_change < 0:
            # Adjust prediction window and SoC thresholds if performance declines
            demand_window = max(1, demand_window - 1)
            base_charge_lb += 0.02
            base_charge_ub += 0.02
        else:
            # Increase prediction window and prioritize availability if performance improves
            demand_window = min(len(customer_arrivals), demand_window + 1)
            base_charge_lb -= 0.01
            base_charge_ub -= 0.01
    algorithm_data['previous_reward'] = reward

    # Dynamically adjust SoC thresholds based on demand variance
    if demand_variance > np.var(customer_arrivals):
        # Increase thresholds to prioritize availability during volatile demand
        base_charge_lb -= 0.05
        base_charge_ub -= 0.05
    else:
        # Decrease thresholds to prioritize charging during stable demand
        base_charge_lb += 0.05
        base_charge_ub += 0.05

    # Implement check for initial step
    if current_step == 0:
        # Prioritize charging for the EV with the lowest SoC
        min_soc_index = np.argmin(battery_soc)
        actions[min_soc_index] = 1

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= base_charge_ub:
            actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Return the operator and updated algorithm data
    return ActionOperator(actions), {'previous_reward': reward}