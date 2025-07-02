from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def demand_responsive_dispatch_15ef(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm for EV fleet charging optimization with adaptive rolling window and feedback loop.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - fleet_size (int): The number of electric vehicles (EVs) in the fleet.
            - total_chargers (int): The maximum number of available chargers.
            - max_time_steps (int): The maximum number of time steps.
            - customer_arrivals (list[int]): A list representing the number of customer arrivals at each time step.
            - charging_price (list[float]): A list representing the charging price at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - operational_status (list[int]): A list indicating the operational status of each EV.
            - battery_soc (list[float]): A list representing the battery state of charge for each EV.
            - time_to_next_availability (list[int]): A list indicating the time until each EV becomes available.
            - current_step (int): The index of the current time step.
        algorithm_data (dict): The algorithm dictionary for current algorithm only. This algorithm does not use this data.
        get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.
        kwargs: Hyper-parameters used in this algorithm. Defaults are set as required.
            - charge_lb (float, default=0.65): Lower bound for battery SoC to consider charging.
            - charge_ub (float, default=0.75): Upper bound for battery SoC to consider charging.
            - base_rolling_window (int, default=5): Base number of time steps to consider for rolling average calculations.

    Returns:
        ActionOperator: An operator that specifies the actions for each EV at the current time step.
        dict: Updated algorithm data (empty in this algorithm).
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    current_step = state_data["current_step"]
    operational_status = state_data["operational_status"]
    battery_soc = state_data["battery_soc"]
    time_to_next_availability = state_data["time_to_next_availability"]

    # Set default hyper-parameters and adjust based on current state
    charge_lb = kwargs.get('charge_lb', 0.65)
    charge_ub = kwargs.get('charge_ub', 0.75)
    base_rolling_window = kwargs.get('base_rolling_window', 5)

    # Calculate demand volatility as the standard deviation of recent customer arrivals
    demand_volatility = np.std(global_data["customer_arrivals"][:current_step]) if current_step > 0 else 0
    adaptive_rolling_window = max(3, int(base_rolling_window * (1 + demand_volatility / 10)))

    # Calculate rolling averages for customer arrivals and charging prices
    if current_step >= adaptive_rolling_window:
        recent_arrivals = global_data["customer_arrivals"][current_step-adaptive_rolling_window:current_step]
        recent_prices = global_data["charging_price"][current_step-adaptive_rolling_window:current_step]
    else:
        recent_arrivals = global_data["customer_arrivals"][:current_step]
        recent_prices = global_data["charging_price"][:current_step]

    rolling_avg_arrivals = np.mean(recent_arrivals) if recent_arrivals else np.mean(global_data["customer_arrivals"])
    rolling_avg_price = np.mean(recent_prices) if recent_prices else np.mean(global_data["charging_price"])

    peak_customer_arrivals = max(global_data["customer_arrivals"])
    fleet_to_charger_ratio = fleet_size / total_chargers

    # Granular adjustment of charge_lb and charge_ub using rolling averages and weighted factors
    if current_step > global_data['max_time_steps'] * 0.5:  # After halfway through the day
        charge_lb = min(charge_lb + (0.05 * (rolling_avg_arrivals / peak_customer_arrivals)), 0.75)
        charge_ub = min(charge_ub + (0.05 * (rolling_avg_price / np.max(global_data["charging_price"]))), 0.80)
    if rolling_avg_price > 0.35:
        charge_lb = max(charge_lb - (0.05 * (rolling_avg_price / np.max(global_data["charging_price"]))), 0.60)
    if rolling_avg_arrivals > 8:
        charge_ub = min(charge_ub + (0.05 * (rolling_avg_arrivals / peak_customer_arrivals)), 0.80)

    # Initialize actions
    actions = [0] * fleet_size

    if fleet_to_charger_ratio > 8:
        # Prioritize charging for EVs with low battery SoC
        # Ensure EVs on a ride remain available
        for i in range(fleet_size):
            if time_to_next_availability[i] >= 1:
                actions[i] = 0
            elif time_to_next_availability[i] == 0 and battery_soc[i] <= charge_lb:
                actions[i] = 1
            elif time_to_next_availability[i] == 0 and battery_soc[i] >= charge_ub:
                actions[i] = 0

        # Ensure the number of charging actions does not exceed available chargers
        actions = [0 if sum(actions) > total_chargers else action for action in actions]

        # Prioritize the EV with the lowest SoC for charging if chargers are available
        if sum(actions) < total_chargers:
            min_soc_ev = min((i for i in range(fleet_size) if time_to_next_availability[i] == 0), key=lambda x: battery_soc[x], default=None)
            if min_soc_ev is not None:
                actions[min_soc_ev] = 1

    # Create and return the ActionOperator
    action_operator = ActionOperator(actions)
    return action_operator, {}