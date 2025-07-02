from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_b6a8(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_B6A8 heuristic algorithm for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - fleet_to_charger_ratio (float): The ratio of fleet size to total chargers, indicating charger scarcity.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        charging_priority_threshold (float, optional): Threshold for prioritizing charging based on SoC. Default is 0.76.
        high_ratio_threshold (float, optional): Threshold for fleet-to-charger ratio to initiate early charging. Default is 15.0.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on SoC levels and charger availability.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    
    # Set default hyper-parameters if not provided
    charging_priority_threshold = kwargs.get("charging_priority_threshold", 0.76)
    high_ratio_threshold = kwargs.get("high_ratio_threshold", 15.0)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # If it's the initial step and the fleet-to-charger ratio is very high
    if current_step == 0 and fleet_size / total_chargers > high_ratio_threshold:
        # Sort EVs by battery SoC in ascending order
        sorted_indices = np.argsort(battery_soc)
        # Select the EV with the lowest SoC for charging
        actions[sorted_indices[0]] = 1

    else:
        # Determine actions for each EV
        for i in range(fleet_size):
            # If EV is serving a trip, it must remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # Prioritize charging for EVs with SoC below the threshold
            elif battery_soc[i] <= charging_priority_threshold:
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

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}