from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_77a7(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm first_charge_77a7 for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dictionary containing global instance information. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): The current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): Algorithm-specific data if necessary.
        get_state_data_function (callable): Function to receive the new solution as input and return the state dictionary for new solution.
        charging_priority_threshold (float, optional): Base threshold for prioritizing charging. Default is 0.6.
        fleet_to_charger_ratio_threshold (float, optional): Threshold for limiting application scope based on fleet-to-charger ratio. Default is 8.0.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on charger availability and dynamic thresholds.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    # Set default hyper-parameters if not provided
    charging_priority_threshold = kwargs.get("charging_priority_threshold", 0.6)
    fleet_to_charger_ratio_threshold = kwargs.get("fleet_to_charger_ratio_threshold", 8.0)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]

    # Prioritize EVs with the lowest SoC for charging when fleet_to_charger_ratio is high
    if fleet_size / total_chargers > fleet_to_charger_ratio_threshold:
        charging_priority_threshold -= 0.1

    # Sort EVs by battery SoC in ascending order
    sorted_ev_indices = sorted(range(fleet_size), key=lambda i: battery_soc[i])

    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in sorted_ev_indices:
        # Ensure EVs on a ride cannot initiate charging
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for the EV with the lowest SoC
        elif battery_soc[i] <= charging_priority_threshold and sum(actions) < total_chargers:
            actions[i] = 1

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}