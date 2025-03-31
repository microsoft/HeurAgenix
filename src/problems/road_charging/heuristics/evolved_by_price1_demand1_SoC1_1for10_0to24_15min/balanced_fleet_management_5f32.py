from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balanced_fleet_management_5f32(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV Fleet Charging Optimization with enhanced early-stage prioritization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution.
        (Optional) charge_lb (float): Base lower bound of SoC for charging decisions. Default is 0.3.
        (Optional) charge_ub (float): Base upper bound of SoC for staying available. Default is 0.4.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Set default hyper-parameters for SoC thresholds
    charge_lb = kwargs.get('charge_lb', 0.3)
    charge_ub = kwargs.get('charge_ub', 0.4)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    order_price = global_data['order_price']

    # Predict future demand and price trends
    if current_step + 5 < len(customer_arrivals):
        future_demand = np.mean(customer_arrivals[current_step:current_step + 5])
    else:
        future_demand = np.mean(customer_arrivals[current_step:])

    # Adjust thresholds based on fleet size and future demand
    if fleet_size > 5:
        charge_lb = 0.5  # Prioritize charging for EVs below 50% SoC
    if future_demand > np.mean(customer_arrivals):
        charge_ub = 0.45  # Keep EVs above 45% SoC available

    # Initialize actions for each EV to zero (remain available)
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0

    # Apply early-stage prioritization if within the first 20 steps
    if current_step < 20:
        # Prioritize charging for EVs that have been idle for consecutive steps
        idle_evs = [i for i in range(fleet_size) if operational_status[i] == 0]
        for ev in idle_evs:
            if battery_soc[ev] < charge_lb and sum(actions) < total_chargers:
                actions[ev] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create and return the ActionOperator with the new actions
    return ActionOperator(actions), {}