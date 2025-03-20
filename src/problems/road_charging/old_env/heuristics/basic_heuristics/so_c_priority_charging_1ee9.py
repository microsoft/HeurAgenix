from src.problems.base.mdp_components import Solution, ActionOperator
from typing import List, Tuple

def so_c_priority_charging_1ee9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> Tuple[ActionOperator, dict]:
    """ Prioritize charging for EVs with the lowest state of charge to ensure they are ready for future rides.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers available.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "battery_soc" (list[float]): Current battery state of charge (SoC) for each vehicle.
            - "ride_lead_time" (list[int]): Remaining time steps for each vehicle currently on a ride.
        (Optional and can be omitted if no algorithm data) algorithm_data (dict): This heuristic does not require any specific algorithm data.
        (Optional and can be omitted if no used) get_state_data_function (callable): Not used in this heuristic.
        (Optional and can be omitted if no hyper parameters data) No hyper-parameters are required for this heuristic.

    Returns:
        ActionOperator: An operator that modifies the solution to prioritize charging for vehicles with the lowest SoC.
        dict: An empty dictionary since this algorithm does not update algorithm data.
    """
    fleet_size = global_data["fleet_size"]
    total_chargers = global_data["total_chargers"]
    battery_soc = state_data["battery_soc"]
    ride_lead_time = state_data["ride_lead_time"]

    # Initialize actions with all 0s, meaning no vehicle is charging initially.
    actions = [0] * fleet_size

    # Collect indices of vehicles that are eligible to charge (not on a ride or fully charged).
    eligible_vehicles = [
        i for i in range(fleet_size)
        if ride_lead_time[i] < 2 and battery_soc[i] < 1
    ]

    # Sort eligible vehicles by their state of charge (SoC) in ascending order.
    eligible_vehicles.sort(key=lambda i: battery_soc[i])

    # Select up to total_chargers vehicles with the lowest SoC to charge.
    for i in eligible_vehicles[:total_chargers]:
        actions[i] = 1

    # Create an ActionOperator with the generated actions.
    operator = ActionOperator(actions)

    # Return the operator and an empty dictionary as no algorithm data is updated.
    return operator, {}