from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_1e23(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_1e23 heuristic algorithm for the Road Charging Problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): The number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): An instance of the Solution class representing the current solution.
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving a trip, 2: charging).
            - unmet_customer_requests (int): The number of customer requests that could not be fulfilled.
        get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for the new solution, ensuring it will not modify the original solution.
        kwargs (dict): Hyper parameters for controlling the algorithm behavior, including:
            - soc_threshold (float, default=0.5): The initial battery state of charge threshold below which charging is considered urgent.
            - customer_arrival_threshold (int, default=10): The initial threshold for average customer arrivals to prioritize dispatch over charging.
            - initial_steps_limit (int, default=10): The number of initial steps where charging is limited to preserve fleet availability.
            - smoothing_factor (float, default=0.1): The factor used to temper adjustments based on historical performance data.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on strategic allocation.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    customer_arrivals = global_data["customer_arrivals"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    operational_status = state_data["operational_status"]
    current_step = state_data["current_step"]
    unmet_customer_requests = state_data.get("unmet_customer_requests", 0)

    # Hyper-parameters with default values
    soc_threshold = kwargs.get("soc_threshold", 0.5)
    customer_arrival_threshold = kwargs.get("customer_arrival_threshold", 10)
    initial_steps_limit = kwargs.get("initial_steps_limit", 10)
    smoothing_factor = kwargs.get("smoothing_factor", 0.1)

    # Calculate average customer arrivals and fleet utilization
    avg_customer_arrivals = np.mean(customer_arrivals)
    fleet_utilization = sum([1 for status in operational_status if status == 1]) / fleet_size

    # Feedback mechanism based on historical performance data
    historical_success_rate = algorithm_data.get("historical_success_rate", 0.75)  # Default to 75% if no data

    # Adjust thresholds using a weighted average approach with smoothing factor
    soc_threshold = soc_threshold * (1 - smoothing_factor) + historical_success_rate * smoothing_factor
    customer_arrival_threshold = customer_arrival_threshold * (1 - smoothing_factor) + avg_customer_arrivals * smoothing_factor

    # Further adjust based on real-time fleet performance metrics
    if unmet_customer_requests > fleet_size * 0.2:  # If unmet requests are high
        soc_threshold = max(soc_threshold - 0.05, 0.1)
        customer_arrival_threshold = max(customer_arrival_threshold - 1, 1)

    # Initialize actions with zeros, indicating no charging by default
    actions = [0] * fleet_size

    # Limit charging actions in initial steps if needed
    if current_step < initial_steps_limit or avg_customer_arrivals > customer_arrival_threshold:
        # Limit charging actions
        return ActionOperator(actions), {}

    # Implement priority queue or round-robin logic for charging
    available_chargers = total_chargers
    for i in range(fleet_size):
        if operational_status[i] == 0 and time_to_next_availability[i] == 0 and battery_soc[i] < soc_threshold:
            if available_chargers > 0:
                actions[i] = 1  # Assign charging action
                available_chargers -= 1

    # Ensure the sum of actions does not exceed the total number of chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size  # Reset to no charging if constraints are violated

    # Return ActionOperator with determined actions and no updates to algorithm data
    return ActionOperator(actions), {"historical_success_rate": historical_success_rate}