from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_ed8f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Heuristic for EV Fleet Charging Optimization in High Competition Scenarios.

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
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. Not used in this algorithm.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the origin solution.
        Introduction for hyper parameters in kwargs if used:
            - base_charge_lb (float, optional): Base lower bound of SoC for charging decisions, dynamically adjusted. Default is 0.5.
            - demand_window (int, optional): Number of time steps to consider for demand prediction. Default is 5.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set default values for hyper-parameters
    base_charge_lb = kwargs.get('base_charge_lb', 0.5)
    demand_window = kwargs.get('demand_window', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']

    # Calculate fleet to charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Predict future demand
    future_demand = np.mean(customer_arrivals[current_step:current_step + demand_window]) if current_step + demand_window < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])
    
    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Sort EVs by battery SoC and time to next availability to prioritize charging for EVs with lowest SoC and shortest availability
    sorted_indices = sorted(range(fleet_size), key=lambda i: (battery_soc[i], time_to_next_availability[i]))

    # Determine actions for each EV
    for i in sorted_indices:
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery SoC and ensure availability for peak demand
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= base_charge_lb:
            actions[i] = 1
            # Limit charging actions to the number of available chargers
            if sum(actions) > total_chargers:
                actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}