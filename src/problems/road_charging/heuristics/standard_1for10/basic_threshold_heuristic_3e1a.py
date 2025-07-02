from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_3e1a(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ A dynamic threshold heuristic for EV Fleet Charging Optimization.

    Args:
        global_data (dict): Contains global instance data. Necessary items are:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - order_price (list[float]): Payment received per minute when a vehicle is on a ride.
        state_data (dict): Contains current state information. Necessary items are:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Contains algorithm-specific data if necessary.
        get_state_data_function (callable): Function to receive the new solution as input and return the state dictionary for new solution.
        kwargs: Hyper-parameters for dynamic threshold control.
            - charge_lb (float, optional): Lower bound for SoC to decide charging. Default is 0.55.
            - charge_ub (float, optional): Upper bound for SoC to stay available. Default is 0.60.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Set hyper-parameters with default values
    charge_lb = kwargs.get('charge_lb', 0.55)
    charge_ub = kwargs.get('charge_ub', 0.60)
    
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    
    # Adjust thresholds based on current step relative to peak customer arrivals
    if current_step < 30:  # Assuming peak hours threshold
        charge_lb = max(charge_lb, 0.55)
        charge_ub = min(charge_ub, 0.65)
    else:
        charge_lb = max(charge_lb, 0.60)
        charge_ub = min(charge_ub, 0.70)

    # Initialize actions based on fleet size
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

    # Sort EVs by battery_soc and prioritize those with lower SoC for charging
    if sum(actions) > total_chargers:
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        charge_indices.sort(key=lambda idx: battery_soc[idx])
        for index in charge_indices[total_chargers:]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}