from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_improvement_ev_charging_c9c9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Greedy Improvement heuristic for the Road Charging Problem.

    This heuristic attempts to enhance the current charging schedule by swapping an EV that is currently charging with an EV that is not, if it results in a higher cumulative reward without violating charger availability constraints.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Ride leading time and length is number of fleet_size.
            - "charging_lead_time" (list[int]): Charging leading time and length is number of fleet_size.
            - "battery_soc" (list[float]): Soc of battery of each fleet and length is number of fleet_size.
            - "reward" (float): The total reward for all fleets for this time step.
        (Optional and can be omitted if no used) get_state_data_function (callable): The function receives the new solution as input and return the state dictionary for new solution, and it will not modify the origin solution.

    Returns:
        ActionOperator: The operator that applies the best swap found.
        dict: An empty dictionary as this heuristic does not update algorithm_data.
    """
    # Extract necessary data
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']
    
    ride_lead_time = state_data['ride_lead_time']
    charging_lead_time = state_data['charging_lead_time']
    battery_soc = state_data['battery_soc']
    current_reward = state_data['reward']

    # Initialize variables
    best_reward_increase = 0
    best_action = [0] * fleet_size  # Default action is to keep all EVs not charging

    # Identify EVs currently charging and not charging
    currently_charging = [i for i in range(fleet_size) if charging_lead_time[i] > 0]
    not_charging = [i for i in range(fleet_size) if charging_lead_time[i] == 0 and ride_lead_time[i] == 0]

    # Evaluate swaps
    for charging_ev in currently_charging:
        for non_charging_ev in not_charging:
            # Calculate potential reward difference
            reward_difference = (charging_price[charging_ev] * battery_soc[charging_ev]) - (charging_price[non_charging_ev] * battery_soc[non_charging_ev])
            
            # Check if the swap results in a reward improvement
            if reward_difference > best_reward_increase:
                best_reward_increase = reward_difference
                # Create a new action with the swap
                best_action = [0] * fleet_size
                best_action[non_charging_ev] = 1  # Schedule the non-charging EV to start charging

    # Return the best action found, if no improvement, return action with all 0
    return ActionOperator(best_action), {}