from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_responsive_dispatch_bc23(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic for EV Fleet Charging Optimization using dynamically adjusted historical weight based on demand variability.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - fleet_to_charger_ratio (float): Ratio of fleet size to total chargers.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        kwargs: Hyper-parameters for controlling algorithm behavior. Defaults are set as required.
            - base_charge_lb (float, optional): Initial lower bound of SoC for charging decisions. Default is 0.75.
            - charging_threshold_adjustment (float, optional): Adjustment factor for charging threshold based on fleet-to-charger ratio. Default is 0.05.
            - variability_weight_factor (float, optional): Factor to adjust historical weight based on demand variability. Default is 0.1.

    Returns:
        ActionOperator: Operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Updated algorithm data or empty dictionary if no update.
    """

    # Set default values for hyper-parameters
    base_charge_lb = kwargs.get('base_charge_lb', 0.75)
    charging_threshold_adjustment = kwargs.get('charging_threshold_adjustment', 0.05)
    variability_weight_factor = kwargs.get('variability_weight_factor', 0.1)

    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']
    customer_arrivals = global_data['customer_arrivals']
    fleet_to_charger_ratio = global_data['fleet_size'] / global_data['total_chargers']

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Calculate demand variability
    if current_step > 1:
        recent_variability = np.std(customer_arrivals[max(0, current_step - 3):current_step + 1])
        historical_variability = np.std(customer_arrivals[:current_step]) if current_step > 0 else recent_variability
        demand_variability = historical_variability - recent_variability
    else:
        demand_variability = 0

    # Adjust historical weight based on demand variability
    historical_weight = 0.5 + variability_weight_factor * demand_variability

    # Calculate the dynamic thresholds using both historical and recent demand patterns
    if current_step > 0:
        recent_demand = np.mean(customer_arrivals[max(0, current_step - 3):current_step + 1])
        historical_demand = np.mean(customer_arrivals[:current_step]) if current_step > 0 else recent_demand
        combined_demand = historical_weight * historical_demand + (1 - historical_weight) * recent_demand
        demand_factor = combined_demand / max(customer_arrivals) if customer_arrivals else 1
    else:
        demand_factor = 1

    charge_lb = base_charge_lb - charging_threshold_adjustment * (fleet_to_charger_ratio / 10) * demand_factor

    # Determine actions for each EV
    for i in range(fleet_size):
        if time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and updated algorithm data
    return ActionOperator(actions), {}