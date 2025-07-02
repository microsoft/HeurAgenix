from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def price_sensitive_charging_10f4(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Heuristic algorithm to schedule charging sessions based on fleet-to-charger ratio, charging prices, and real-time demand fluctuations with a scaling factor.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): The maximum number of available chargers.
            - "charging_price" (list[float]): A list representing the charging price at each time step.
            - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
            - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "time_to_next_availability" (list[int]): Lead time until the fleet becomes available.
            - "battery_soc" (list[float]): Battery state of charge in percentage.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): Not used in this algorithm.
        (Optional and can be omitted if no used) get_state_data_function (callable): Not used in this algorithm.
        Introduction for hyper parameters in kwargs if used:
            - "base_charging_priority_threshold" (float, default=0.05): The base threshold to prioritize EVs based on their SoC relative to the fleet average.
            - "demand_scaling_factor" (float, default=1.0): A scaling factor for adjusting the influence of customer arrival rates on the dynamic threshold.

    Returns:
        ActionOperator: Operator defining new actions for EVs at the current time step.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary data
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"]
    current_step = state_data["current_step"]
    time_to_next_availability = state_data["time_to_next_availability"]
    battery_soc = state_data["battery_soc"]
    fleet_size = global_data["fleet_size"]
    customer_arrivals = global_data["customer_arrivals"]
    
    # Hyper-parameters
    base_charging_priority_threshold = kwargs.get("base_charging_priority_threshold", 0.05)
    demand_scaling_factor = kwargs.get("demand_scaling_factor", 1.0)
    
    # Calculate variance of battery SoC to adjust the threshold dynamically
    soc_variance = np.var(battery_soc) if len(battery_soc) > 0 else 0
    
    # Adjust threshold based on real-time customer demand with scaling factor
    demand_factor = customer_arrivals[current_step] / np.mean(customer_arrivals) if np.mean(customer_arrivals) > 0 else 1
    dynamic_threshold = base_charging_priority_threshold * (1 + soc_variance) * demand_factor * demand_scaling_factor

    # Initialize actions with zeros
    actions = [0] * fleet_size
    
    # Check if charging_price data is available
    if not charging_price:
        # If no charging price data, return an action operator with all zeros
        return ActionOperator(actions), {}

    # Calculate the average SoC of the fleet
    avg_soc = np.mean(battery_soc) if len(battery_soc) > 0 else 0
    
    # Calculate fleet-to-charger ratio
    fleet_to_charger_ratio = fleet_size / total_chargers if total_chargers > 0 else float('inf')

    # Sort EVs by battery SoC close to the average SoC for prioritization
    ev_indices = np.argsort(np.abs(np.array(battery_soc) - avg_soc))
    
    # Charging logic
    chargers_used = 0
    for i in ev_indices:
        if time_to_next_availability[i] > 0:
            # EV is currently on a ride, cannot charge
            actions[i] = 0
        elif chargers_used < total_chargers and charging_price[current_step] <= np.mean(charging_price) and fleet_to_charger_ratio > 10:
            # Assign charging action if chargers are available, current price is low, and fleet-to-charger ratio is high
            if np.abs(battery_soc[i] - avg_soc) <= dynamic_threshold:
                actions[i] = 1
                chargers_used += 1
        else:
            # No more chargers available or conditions not met, remaining EVs stay idle
            break

    # Ensure at least one EV is set to charge if possible
    if chargers_used == 0 and charging_price[current_step] <= np.mean(charging_price):
        for i in ev_indices:
            if time_to_next_availability[i] == 0 and chargers_used < total_chargers:
                actions[i] = 1
                chargers_used += 1
                break

    return ActionOperator(actions), {}