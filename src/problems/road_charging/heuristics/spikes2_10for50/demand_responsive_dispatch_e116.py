from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_responsive_dispatch_e116(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm with granular dynamic adjustment, considering historical trends and predictive analytics for customer arrivals and charging prices.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - charging_price (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV.
            - battery_soc (list[float]): A list representing the battery state of charge for each EV.
            - time_to_next_availability (list[int]): A list indicating the time until each EV becomes available.
            - current_step (int): The index of the current time step.
        algorithm_data (dict): Not necessary for this algorithm.
        get_state_data_function (callable): Function that receives the new solution and returns the state dictionary for the new solution.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step.
        dict: Updated algorithm data. Empty in this case.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    charging_price = global_data["charging_price"]
    
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]
    current_step = state_data["current_step"]

    # Initialize action list for all EVs
    actions = [0] * fleet_size

    # Use moving average for customer arrivals and charging prices
    window_size = 5  # Define the window size for moving average
    avg_customer_arrivals = np.mean(customer_arrivals[max(0, current_step-window_size+1):current_step+1])
    avg_charging_price = np.mean(charging_price[max(0, current_step-window_size+1):current_step+1])

    # Determine dynamic SoC threshold based on historical trends
    dynamic_soc_threshold = 0.3  # Default base threshold
    peak_arrival_threshold = avg_customer_arrivals * 0.2  # Dynamic adjustment based on average customer arrivals
    high_price_threshold = avg_charging_price * 0.5  # Dynamic adjustment based on average charging prices

    if avg_customer_arrivals > peak_arrival_threshold:
        dynamic_soc_threshold += 0.1  # Increase threshold during high average customer demand
    if avg_charging_price > high_price_threshold:
        dynamic_soc_threshold -= 0.1  # Decrease threshold during high average charging prices

    # Prioritize EVs that are currently serving rides and have low SoC, and are about to become available
    prioritize_for_charging = [i for i in range(fleet_size) if operational_status[i] == 1 and time_to_next_availability[i] == 0 and battery_soc[i] < dynamic_soc_threshold]

    # List of idle EVs eligible for charging
    eligible_evs = [i for i in range(fleet_size) if operational_status[i] == 0 and time_to_next_availability[i] == 0 and i not in prioritize_for_charging]

    # Combine prioritized EVs with eligible EVs
    prioritized_evs = prioritize_for_charging + eligible_evs

    # Sort prioritized EVs based on SoC in ascending order (prioritize low SoC for charging)
    prioritized_evs.sort(key=lambda i: battery_soc[i])

    # Assign charging actions up to the number of available chargers
    for i in range(min(len(prioritized_evs), total_chargers)):
        actions[prioritized_evs[i]] = 1

    # Ensure no EV serving a ride is assigned a charging action
    actions = [0 if time_to_next_availability[i] > 0 else actions[i] for i in range(fleet_size)]

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, {}