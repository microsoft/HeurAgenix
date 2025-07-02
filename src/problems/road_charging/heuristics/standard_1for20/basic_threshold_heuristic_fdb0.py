from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_fdb0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Threshold Heuristic with Real-Time Adjustments for EV Fleet Charging Optimization.

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
        algorithm_data (dict): Not used in this algorithm.
        get_state_data_function (callable): Not used directly, placeholder for function interface.
        introduction for hyper parameters in kwargs if used.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']

    # Dynamic hyper-parameter adjustment based on real-time data
    future_demand = np.mean(customer_arrivals[current_step:current_step + 5]) if current_step + 5 < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])
    base_charge_lb = np.percentile(battery_soc, 25)  # Lower bound based on current SoC distribution
    base_charge_ub = np.percentile(battery_soc, 50)  # Upper bound set as median

    # Calculate fleet to charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Prioritize charging for EVs with real-time SoC thresholds and idle status during high competition
    if fleet_to_charger_ratio > 15.0:
        # Sort EVs by battery SoC to prioritize charging for EVs with lowest SoC
        sorted_indices = sorted(range(fleet_size), key=lambda i: battery_soc[i])
        
        # Determine actions for each EV
        for i in sorted_indices:
            # If EV is serving a trip, it must remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # Prioritize charging for EVs with SoC just above dynamically adjusted base_charge_lb
            elif time_to_next_availability[i] == 0 and base_charge_lb <= battery_soc[i] <= base_charge_ub:
                actions[i] = 1

        # Ensure the sum of actions does not exceed the total number of chargers
        if sum(actions) > total_chargers:
            excess_count = sum(actions) - total_chargers
            charge_indices = [index for index, action in enumerate(actions) if action == 1]
            np.random.shuffle(charge_indices)
            for index in charge_indices[:excess_count]:
                actions[index] = 0

    # If fleet-to-charger ratio is not high, maintain availability
    else:
        actions = [0] * fleet_size

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}