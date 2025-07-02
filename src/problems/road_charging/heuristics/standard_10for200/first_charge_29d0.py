from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_29d0(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_29d0 heuristic algorithm for EV Fleet Charging Optimization.

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
        (Optional) algorithm_data (dict): The algorithm dictionary for current algorithm only.
        (Optional) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        (Optional) introduction for hyper parameters in kwargs:
            - charging_priority_threshold (float, optional): Base threshold for prioritizing charging. Default is 0.6.
            - fleet_to_charger_ratio_threshold (float, optional): Threshold for limiting application scope based on fleet-to-charger ratio. Default is 8.0.
            - decay_factor (float, optional): Initial factor for temporal decay of success rates. Default is 0.9.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on charger availability and dynamic thresholds.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    
    # Set default hyper-parameters if not provided
    charging_priority_threshold = kwargs.get("charging_priority_threshold", 0.6)
    fleet_to_charger_ratio_threshold = kwargs.get("fleet_to_charger_ratio_threshold", 8.0)
    decay_factor = kwargs.get("decay_factor", 0.9)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]
    order_price = global_data["order_price"]

    # Calculate rolling average and variance of past demand
    demand_window = 5
    past_demand = customer_arrivals[max(0, current_step - demand_window):current_step]
    average_past_demand = np.mean(past_demand) if past_demand else 0
    demand_variance = np.var(past_demand) if len(past_demand) > 1 else 0

    # Dynamic adjustment of charging priority threshold based on variance
    future_demand = np.mean(customer_arrivals[current_step:current_step + demand_window]) if current_step + demand_window < len(customer_arrivals) else np.mean(customer_arrivals[current_step:])
    peak_customer_arrivals = np.max(customer_arrivals)
    
    if fleet_size > fleet_to_charger_ratio_threshold * total_chargers:
        charging_priority_threshold += 0.1 * (future_demand / peak_customer_arrivals)
    
    charging_priority_threshold *= (1 + demand_variance / (average_past_demand + 1e-5))

    # Learning component to adapt the decay factor based on historical success rates
    individual_success_rates = algorithm_data.get('individual_success_rates', [0.0] * fleet_size)
    historical_success_factor = np.mean(individual_success_rates) if individual_success_rates else 0
    decay_factor = max(0.5, min(decay_factor * (1 + historical_success_factor), 1.0))

    # Granular feedback loop with temporal decay for individual EV success rates
    for i in range(fleet_size):
        if battery_soc[i] < charging_priority_threshold:
            charging_priority_threshold *= (1 + individual_success_rates[i])
        # Apply temporal decay
        individual_success_rates[i] *= decay_factor

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # List EVs by priority based on SoC and operational status
    ev_priority = sorted(
        [(i, battery_soc[i], operational_status[i]) for i in range(fleet_size)],
        key=lambda x: (x[2] == 0, x[1])
    )

    # Determine actions for each EV
    chargers_used = 0
    for i, soc, status in ev_priority:
        if time_to_next_availability[i] >= 1:
            continue  # Cannot charge while serving a trip
        if chargers_used < total_chargers and soc <= charging_priority_threshold:
            actions[i] = 1
            chargers_used += 1

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Update individual success rates
    for i in range(fleet_size):
        if actions[i] == 1:
            individual_success_rates[i] += 1

    # Normalize success rates
    individual_success_rates = [rate / max(1, fleet_size) for rate in individual_success_rates]

    # Return the operator and updated algorithm data
    return ActionOperator(actions), {'individual_success_rates': individual_success_rates}