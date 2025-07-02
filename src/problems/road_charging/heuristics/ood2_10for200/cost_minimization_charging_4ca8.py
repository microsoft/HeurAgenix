from src.problems.base.mdp_components import ActionOperator
import numpy as np

def cost_minimization_charging_4ca8(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Dynamic Charging Optimization for EV Fleet with Real-Time Adjustments.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - charging_rate (list[float]): List of charging rates at each time step.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        Hyper-parameters in kwargs:
            - window_size (int, optional): Size of the sliding window for average calculations, default is 5.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """
    
    # Set default hyper-parameters if not provided
    window_size = kwargs.get('window_size', 5)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']

    # Calculate dynamic thresholds using weighted averages
    recent_arrivals = customer_arrivals[max(0, current_step - window_size):current_step + 1]
    average_customer_arrivals = np.mean(recent_arrivals) if recent_arrivals else np.mean(customer_arrivals)
    
    soc_threshold = np.percentile(battery_soc, 50)  # Median SoC as a dynamic threshold

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Dynamic charging strategy based on real-time metrics
    for i in range(fleet_size):
        if time_to_next_availability[i] >= 1:  # Vehicle on a ride
            actions[i] = 0
        elif battery_soc[i] < soc_threshold and sum(actions) < total_chargers:
            actions[i] = 1
        elif average_customer_arrivals > np.percentile(customer_arrivals, 75):  # High demand
            if battery_soc[i] < soc_threshold * 1.1 and sum(actions) < total_chargers:
                actions[i] = 1

    # Return an action operator with determined actions, ensuring compliance with constraints
    return ActionOperator(actions), {}