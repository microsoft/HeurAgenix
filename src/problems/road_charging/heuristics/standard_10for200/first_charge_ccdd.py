from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_ccdd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_ccdd heuristic algorithm for EV Fleet Charging Optimization with dynamic threshold adjustment.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - order_price (list[float]): A list representing the payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        charging_priority_threshold (float, optional): Base threshold for prioritizing charging based on SoC. Default is 0.70.
        fleet_to_charger_ratio_threshold (float, optional): Threshold for fleet-to-charger ratio to initiate prioritization logic. Default is 10.0.
        high_demand_multiplier (float, optional): Multiplier to adjust charging priority threshold during high demand. Default is 0.90.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on dynamically adjusted SoC levels and charger availability.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    
    # Set default hyper-parameters if not provided
    charging_priority_threshold = kwargs.get("charging_priority_threshold", 0.70)
    fleet_to_charger_ratio_threshold = kwargs.get("fleet_to_charger_ratio_threshold", 10.0)
    high_demand_multiplier = kwargs.get("high_demand_multiplier", 0.90)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]

    # Dynamic adjustment of charging priority threshold based on demand and order price
    if customer_arrivals[current_step] > np.mean(customer_arrivals) and order_price[current_step] > np.mean(order_price):
        dynamic_threshold = charging_priority_threshold * high_demand_multiplier
    else:
        dynamic_threshold = charging_priority_threshold

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Apply prioritization logic only when fleet-to-charger ratio is high
    if fleet_size / total_chargers > fleet_to_charger_ratio_threshold:
        # Determine actions for each EV
        for i in range(fleet_size):
            # If EV is serving a trip, it must remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # Prioritize charging for EVs with low battery_soc and are idle
            elif operational_status[i] == 0 and battery_soc[i] <= dynamic_threshold:
                actions[i] = 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        charge_indices.sort(key=lambda idx: (battery_soc[idx], time_to_next_availability[idx]))
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}