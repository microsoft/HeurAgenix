from src.problems.base.mdp_components import Solution, ActionOperator
import numpy as np

def greedy_charging_schedule_ade9(global_data: dict, state_data: dict, algorithm_data: dict, get_state_data_function: callable, **kwargs) -> tuple[ActionOperator, dict]:
    """Greedy Charging Schedule heuristic for the road_charging problem.

    This algorithm attempts to construct an optimal charging schedule by iteratively selecting the EV that should charge at a given time step to minimize immediate costs or maximize immediate rewards.
    It starts with no EVs scheduled to charge and considers the available charging slots, selecting the EV that can charge at the lowest cost or highest reward based on the current state and constraints.
    The heuristic considers the state of charge, charging rates, and current charging prices to make decisions, ensuring that capacity constraints and fairness are respected.
    This process repeats until no further charging actions can be scheduled without violating constraints or all EVs have reached their required state of charge.

    Args:
        global_data (dict): The global data dict containing the global data. In this algorithm, the following items are necessary:
            - "fleet_size" (int): Number of EVs in the fleet.
            - "max_time_steps" (int): Maximum number of time steps.
            - "total_chargers" (int): Total number of chargers.
            - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step.
            - "order_price" (list[float]): Payments (in dollars) received per time step when a vehicle is on a ride.
            - "consume_rate" (list[float]): Battery consumption rate per time step for each vehicle.
            - "charging_rate" (list[float]): Battery charging rate per time step for each vehicle.
            - "initial_charging_cost" (float): Cost incurred for the first connection to a charger.
        
        state_data (dict): The state dictionary containing the current state information. In this algorithm, the following items are necessary:
            - "ride_lead_time" (list[int]): Remaining ride time for each EV.
            - "battery_soc" (list[float]): State of charge of each EV's battery.
            - "charging_lead_time" (list[int]): Charging time steps already used by each EV.

    Returns:
        ActionOperator: The operator that schedules EVs to charge at the current time step.
        dict: Empty dictionary as the algorithm does not update any algorithm-specific data.
    """
    fleet_size = global_data['fleet_size']
    total_chargers = global_data['total_chargers']
    charging_price = global_data['charging_price']
    order_price = global_data['order_price']
    consume_rate = global_data['consume_rate']
    charging_rate = global_data['charging_rate']
    initial_charging_cost = global_data['initial_charging_cost']
    
    ride_lead_time = state_data['ride_lead_time']
    battery_soc = state_data['battery_soc']
    charging_lead_time = state_data['charging_lead_time']

    # Initialize actions with no charging (0) for each EV
    actions = [0] * fleet_size

    # Calculate potential costs or rewards for charging each EV
    potential_costs = []
    for i in range(fleet_size):
        if ride_lead_time[i] > 0:
            potential_costs.append(float('inf'))  # If on ride, set high cost to avoid charging
        else:
            cost = initial_charging_cost if charging_lead_time[i] == 0 else 0
            cost += charging_price[0] * charging_rate[i] - order_price[0] * consume_rate[i]
            potential_costs.append(cost)

    # Select EVs to charge based on lowest cost and available chargers
    sorted_ev_indices = np.argsort(potential_costs)
    chargers_used = 0

    for idx in sorted_ev_indices:
        if chargers_used < total_chargers and potential_costs[idx] < float('inf'):
            actions[idx] = 1
            chargers_used += 1

    return ActionOperator(actions), {}