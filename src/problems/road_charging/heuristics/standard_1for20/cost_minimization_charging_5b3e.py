from src.problems.base.mdp_components import *
import numpy as np

def cost_minimization_charging_5b3e(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Stabilized Adaptive Strategy.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - charging_rate (list[float]): List of charging rates at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        Hyper-parameters in kwargs:
            - initial_steps (int, optional): Number of initial steps to use conservative strategy, default is 5.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set default hyper-parameters if not provided
    initial_steps = kwargs.get('initial_steps', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']

    # Calculate dynamic thresholds based on real-time fleet metrics
    average_soc = np.mean(battery_soc)
    utilization_rate = sum(1 for status in operational_status if status != 0) / fleet_size
    base_charge_lb = max(0.65, min(0.75, 1 - utilization_rate))  # Bound between 0.65 and 0.75
    base_charge_ub = max(0.75, min(0.85, average_soc + 0.05))  # Bound between 0.75 and 0.85

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # If within initial conservative steps, set all actions to 0 if average battery_soc is above 0.75
    if current_step < initial_steps and average_soc > 0.75:
        return ActionOperator(actions), {}

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure vehicles on a ride cannot charge
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize maintaining availability for EVs with battery_soc above base_charge_lb
        elif battery_soc[i] > base_charge_lb:
            actions[i] = 0
        # Allow charging for EVs with battery_soc below base_charge_lb, ensuring chargers are not exceeded
        elif battery_soc[i] <= base_charge_lb and sum(actions) < total_chargers:
            actions[i] = 1

    # Return an action operator with determined actions, ensuring compliance with constraints
    return ActionOperator(actions), {}