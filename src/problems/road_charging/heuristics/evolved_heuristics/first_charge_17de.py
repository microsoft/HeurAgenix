from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_17de(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_17de heuristic algorithm for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - order_price (list[float]): List of payment received per minute when a vehicle is on a ride.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
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
    order_price = global_data["order_price"]

    # Predict future demand and price trends
    future_demand = np.mean(customer_arrivals[current_step:current_step + 5]) if current_step + 5 < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])
    future_order_price = np.mean(order_price[current_step:current_step + 5]) if current_step + 5 < len(order_price) else np.mean(order_price[current_step:])
    peak_customer_arrivals = np.max(customer_arrivals)

    # Adjust thresholds based on future demand and price trends
    if fleet_size > fleet_to_charger_ratio_threshold * total_chargers:
        charging_priority_threshold = charging_priority_threshold + 0.1 * (future_demand / peak_customer_arrivals)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # If EV is serving a trip, it must remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for EVs with low battery_soc when future demand is high
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charging_priority_threshold:
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