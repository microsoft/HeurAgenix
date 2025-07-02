from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_b8eb(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ A refined heuristic for EV Fleet Charging Optimization based on dynamic thresholds.

    Args:
        global_data (dict): Contains global instance data. Necessary items are:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): Contains current state information. Necessary items are:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        kwargs: Hyper-parameters for dynamic threshold control.
            - charge_lb (float, optional): Lower bound for SoC to decide charging. Default is 0.35.
            - charge_ub (float, optional): Upper bound for SoC to stay available. Default is 0.65.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Set hyper-parameters with default values
    charge_lb = kwargs.get('charge_lb', 0.35)
    charge_ub = kwargs.get('charge_ub', 0.65)
    
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    
    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Calculate average customer arrivals for peak hour determination
    average_customer_arrivals = np.mean(customer_arrivals)

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Adjust threshold logic based on peak hours
        elif customer_arrivals[current_step] > average_customer_arrivals:
            if battery_soc[i] < charge_lb:
                actions[i] = 1
            elif battery_soc[i] > charge_ub:
                actions[i] = 0
        else:
            # Off-peak charging prioritization
            if battery_soc[i] < charge_ub:
                actions[i] = 1
            else:
                actions[i] = 0
    
    # Ensure the sum of actions doesn't exceed the number of chargers
    if sum(actions) > total_chargers:
        # Sort EVs by battery_soc and prioritize those with lower SoC for charging
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        charge_indices.sort(key=lambda idx: battery_soc[idx])
        for index in charge_indices[total_chargers:]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}