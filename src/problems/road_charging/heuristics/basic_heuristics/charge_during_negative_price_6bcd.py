from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def charge_during_negative_price_6bcd(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """ Schedule charging when the charging price is negative to minimize costs and potentially earn from charging.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "total_chargers" (int): Total number of chargers available.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step. 
            - "ride_lead_time" (list[int]): Ride lead time for each vehicle in the fleet.
            - "battery_soc" (list[float]): State of charge (SoC) for each vehicle in the fleet.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and returns the state dictionary for new solution, and it will not modify the original solution.

    Returns:
        ActionOperator: An operator that sets actions for charging EVs when the charging price is negative, respecting the constraints.
        dict: An empty dictionary as no updates to algorithm data are performed.
    """
    # Extract necessary information from global_data and state_data
    current_step = state_data["current_step"]
    total_chargers = global_data["total_chargers"]
    charging_price = global_data["charging_price"][current_step]
    ride_lead_time = state_data["ride_lead_time"]
    battery_soc = state_data["battery_soc"]

    # Initialize actions for the fleet
    actions = [0] * len(ride_lead_time)

    # If the charging price is negative, consider scheduling charging
    if charging_price < 0:
        # Iterate over each vehicle to determine if it should charge
        for i in range(len(ride_lead_time)):
            # Check the constraints for charging
            if ride_lead_time[i] < 2 and battery_soc[i] < 1:
                actions[i] = 1

        # Ensure the sum of actions does not exceed the total chargers available
        if sum(actions) > total_chargers:
            # Prioritize charging based on some criteria, e.g., lowest SoC
            soc_sorted_indices = np.argsort(battery_soc)
            actions = [0] * len(actions)
            for idx in soc_sorted_indices:
                if ride_lead_time[idx] < 2 and battery_soc[idx] < 1:
                    actions[idx] = 1
                    if sum(actions) == total_chargers:
                        break

    # Create an ActionOperator with the determined actions
    operator = ActionOperator(actions)

    # Return the operator and an empty dictionary for algorithm data
    return operator, {}