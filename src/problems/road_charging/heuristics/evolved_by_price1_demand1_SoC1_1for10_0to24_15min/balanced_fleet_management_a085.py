from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def balanced_fleet_management_a085(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV Fleet Charging Optimization with real-time SoC analysis.

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
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, the following items are necessary:
            - necessary_key (value type): description
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        Introduction for hyper parameters in kwargs:
            - charge_lb (float): Lower bound of SoC for charging decisions. Default is 0.60.
            - charge_ub (float): Upper bound of SoC for staying available. Default is 0.65.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Set default hyper-parameters for SoC thresholds
    charge_lb = kwargs.get('charge_lb', 0.60)
    charge_ub = kwargs.get('charge_ub', 0.65)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    order_price = global_data['order_price']

    # Dynamic adjustment of thresholds based on current step and demand patterns
    avg_customer_arrivals = np.mean(customer_arrivals)
    if current_step > 0.5 * len(customer_arrivals):
        charge_lb += 0.05
        charge_ub += 0.05

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

    # Prioritize charging for idle EVs when fleet_to_charger_ratio is high
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create and return the ActionOperator with the new actions
    return ActionOperator(actions), {}