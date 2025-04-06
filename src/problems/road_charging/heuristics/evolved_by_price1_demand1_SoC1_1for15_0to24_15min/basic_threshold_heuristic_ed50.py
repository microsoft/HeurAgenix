from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def basic_threshold_heuristic_ed50(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Dynamic Threshold Heuristic with Weighted Real-Time Adjustments for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): Number of electric vehicles in the fleet.
            - total_chargers (int): Maximum number of available chargers.
            - customer_arrivals (list[int]): Number of customer arrivals at each time step.
            - charging_price (list[float]): Charging price at each time step.
            - order_price (list[float]): Order price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): The algorithm dictionary for current algorithm only. In this algorithm, no specific items are necessary.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator: An operator with actions for the EV fleet, ensuring actions comply with constraints.
        dict: Empty dictionary as no algorithm data is updated in this heuristic.
    """
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    customer_arrivals = global_data['customer_arrivals']
    charging_price = global_data['charging_price']
    order_price = global_data['order_price']
    current_step = state_data['current_step']
    operational_status = state_data['operational_status']
    time_to_next_availability = state_data['time_to_next_availability']
    battery_soc = state_data['battery_soc']

    # Calculate averages
    avg_customer_arrivals = np.mean(customer_arrivals)
    avg_charging_price = np.mean(charging_price)
    avg_order_price = np.mean(order_price)

    # Dynamically adjust charge_lb and charge_ub using weighted factors
    charge_lb = 0.5 + 0.1 * (avg_customer_arrivals / 10) - 0.05 * (avg_charging_price / 0.3) + 0.05 * (avg_order_price / 1.0)
    charge_ub = 0.4 + 0.05 * (avg_customer_arrivals / 10) - 0.03 * (avg_charging_price / 0.3) + 0.02 * (avg_order_price / 1.0)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Prioritize initial charging step
    if current_step == 0:
        average_soc = np.mean(battery_soc)
        if average_soc > 0.75:
            # Charge the EV with the highest SoC
            max_soc_index = np.argmax(battery_soc)
            actions[max_soc_index] = 1

    else:
        # Determine actions for each EV
        for i in range(fleet_size):
            # If EV is serving a trip, it must remain available
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            # If EV is idle and SoC is below the dynamically adjusted lower bound, attempt to charge
            elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
                actions[i] = 1
            # If EV is idle and SoC is above the dynamically adjusted upper bound, remain available
            elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
                actions[i] = 0

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        # Randomly set excess charging actions to 0 to comply with charger constraints
        excess_count = sum(actions) - total_chargers
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        np.random.shuffle(charge_indices)
        for index in charge_indices[:excess_count]:
            actions[index] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}