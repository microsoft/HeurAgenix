from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def min_cost_charging_insertion_1599(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Min-Cost Charging Insertion heuristic for the road_charging problem.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "current_step" (int): The index of the current time step.
            - "ride_lead_time" (list[int]): Ride leading time for each vehicle.
            - "battery_soc" (list[float]): State of charge for each vehicle.
        (Optional) algorithm_data (dict): The algorithm dictionary for current algorithm only. 
        (Optional) get_state_data_function (callable): A function to retrieve the state data of a new solution.
        kwargs: Contains hyper-parameters such as:
            - "cost_threshold" (float): The threshold below which a charging action is considered cost-effective. Default is 0.

    Returns:
        ActionOperator: The operator to apply charging actions for vehicles.
        dict: An empty dictionary since this algorithm doesn't update the algorithm data.
    """
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']
    current_step = state_data['current_step']
    ride_lead_time = state_data['ride_lead_time']
    battery_soc = state_data['battery_soc']
    cost_threshold = kwargs.get('cost_threshold', 0)

    # Initialize actions for all vehicles to 0 (no charging)
    actions = [0] * fleet_size

    # Track available chargers
    available_chargers = total_chargers

    # Iterate over each vehicle to decide on charging actions
    for i in range(fleet_size):
        if ride_lead_time[i] >= 2 or battery_soc[i] >= 1:
            # Vehicle is on a ride or fully charged, cannot charge
            continue

        # Check if charging is possible at this time step
        if available_chargers > 0 and charging_price[current_step] < np.inf:
            # Compute cost efficiency for charging at this step
            cost_efficiency = charging_price[current_step]

            # Decide to charge if it is cost-efficient
            if cost_efficiency < cost_threshold:
                actions[i] = 1
                available_chargers -= 1

    # Ensure the number of charging actions does not exceed available chargers
    if sum(actions) > total_chargers:
        actions = [0] * fleet_size

    return ActionOperator(actions), {}