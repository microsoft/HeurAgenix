from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def first_charge_231f(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ FirstCharge_231f heuristic algorithm for EV Fleet Charging Optimization.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - customer_arrivals (list[int]): List of customer arrivals at each time step.
            - max_time_steps (int): The maximum number of time steps.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - current_solution (Solution): Current action trajectory for EVs.
            - current_step (int): Index of the current time step.
            - operational_status (list[int]): Operational status of each EV (0: idle, 1: serving, 2: charging).
            - time_to_next_availability (list[int]): Lead time until each EV becomes available.
            - battery_soc (list[float]): Battery state of charge for each EV in percentage.
        algorithm_data (dict): The algorithm dictionary for current algorithm only if necessary.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution.
        kwargs: Hyper-parameters used by the algorithm:
            - charge_lb (float, optional): Lower bound for charging priority. Default is 0.60.
            - charge_ub (float, optional): Upper bound for charging priority. Default is 0.65.
            - fleet_to_charger_ratio_threshold (float, optional): Threshold for limiting application scope based on fleet-to-charger ratio. Default is 8.0.

    Returns:
        An ActionOperator that modifies the solution to assign charging actions to EVs based on charger availability and dynamic thresholds.
        An empty dictionary as this algorithm does not update algorithm data.
    """
    
    # Set default hyper-parameters if not provided
    charge_lb = kwargs.get('charge_lb', 0.60)
    charge_ub = kwargs.get('charge_ub', 0.65)
    fleet_to_charger_ratio_threshold = kwargs.get('fleet_to_charger_ratio_threshold', 8.0)

    # Extract necessary data
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    customer_arrivals = global_data["customer_arrivals"]
    max_time_steps = global_data["max_time_steps"]
    
    # Predict future demand variance
    future_demand_variance = np.var(customer_arrivals[current_step:min(current_step + 5, max_time_steps)]) if customer_arrivals else 0
    peak_customer_arrivals = np.max(customer_arrivals) if customer_arrivals else 0

    # Introduce a scaling factor based on demand variance
    scaling_factor = 1 + 0.1 * (future_demand_variance / peak_customer_arrivals if peak_customer_arrivals > 0 else 0)
    charge_lb *= scaling_factor
    charge_ub *= scaling_factor

    # Adjust charge_lb based on fleet-to-charger ratio
    if fleet_size > fleet_to_charger_ratio_threshold * total_chargers:
        charge_lb = max(charge_lb - 0.05 * (np.mean(customer_arrivals[current_step:min(current_step + 5, max_time_steps)]) / peak_customer_arrivals), 0)

    # Initialize actions based on fleet size
    actions = [0] * fleet_size

    # Determine actions for each EV
    for i in range(fleet_size):
        # Ensure EVs on a ride remain available
        if time_to_next_availability[i] >= 1:
            actions[i] = 0
        # Prioritize charging for idle EVs with low battery SoC
        elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
            actions[i] = 1
        elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
            actions[i] = 0
    
    # Prioritize EV with lowest battery_soc when multiple EVs have similar levels
    if sum(actions) > total_chargers:
        charge_indices = [index for index, action in enumerate(actions) if action == 1]
        sorted_indices = sorted(charge_indices, key=lambda idx: battery_soc[idx])
        for idx in sorted_indices[total_chargers:]:
            actions[idx] = 0

    # Create a new solution with the determined actions
    new_solution = Solution([actions])

    # Return the operator and empty algorithm data
    return ActionOperator(actions), {}