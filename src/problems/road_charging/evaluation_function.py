# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

def get_global_data_feature(global_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): The global data dict containing the global instance data with:
    - "fleet_size" (int): Number of EVs in the fleet.
    - "max_time_steps" (int): Maximum number of time steps, e.g., 5 minutes as a time step, 24 hours corresponds to 288 time steps.
    - "total_chargers" (int): Total number of chargers.
    - "max_cap" (int): Max battery capacity in kWh.
    - "consume_rate" (list[float]): Battery consumption rate (as a percentage) per time step for each vehicle in the fleet. Length is fleet_size.
    - "charging_rate" (list[float]): Battery charging rate (as a percentage) per time step for each vehicle in the fleet. Length is fleet_size.
    - "assign_prob" (float): Probability of receiving a ride order when the vehicle is in idle status.
    - "order_price" (list[float]): Payments (in dollars) received per time step when a vehicle is on a ride. The payment is assumed to be the same for all vehicles. The length of the list is max_time_steps.
    - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step. The price is assumed to be the same for all vehicles. The length of the list is max_time_steps.
    - "initial_charging_cost" (float): Cost (in dollars) incurred for the first connection to a charger.

    Returns:
        dict: The feature of global data, which can represents the global data with:
            - average_consume_rate (float): Average battery consumption rate across the fleet.
            - average_charging_rate (float): Average battery charging rate across the fleet.
            - total_energy_demand (int): Total potential energy demand if all vehicles need a full charge.
            - charger_to_vehicle_ratio (float): Ratio of chargers to vehicles.
            - average_order_price (float): Average payment received per time step from ride orders.
            - average_charging_price (float): Average charging price per kWh across all time steps.
            - peak_charging_price (float): Highest charging price per kWh across all time steps.
            - average_assign_prob (float): Probability of receiving a ride order when idle.
    """
    # Calculate the average consumption rate across the fleet
    average_consume_rate = sum(global_data['consume_rate']) / global_data['fleet_size']
    
    # Calculate the average charging rate across the fleet
    average_charging_rate = sum(global_data['charging_rate']) / global_data['fleet_size']
    
    # Calculate the total energy demand if all vehicles need a full charge
    total_energy_demand = global_data['fleet_size'] * global_data['max_cap']
    
    # Calculate the ratio of chargers to vehicles
    charger_to_vehicle_ratio = global_data['total_chargers'] / global_data['fleet_size']
    
    # Calculate the average order price per time step
    average_order_price = sum(global_data['order_price']) / len(global_data['order_price'])
    
    # Calculate the average charging price per kWh
    average_charging_price = sum(global_data['charging_price']) / len(global_data['charging_price'])
    
    # Identify the peak charging price per kWh
    peak_charging_price = max(global_data['charging_price'])
    
    # Assign the probability of receiving a ride order when idle
    average_assign_prob = global_data['assign_prob']
    
    return {
        'average_consume_rate': average_consume_rate,
        'average_charging_rate': average_charging_rate,
        'total_energy_demand': total_energy_demand,
        'charger_to_vehicle_ratio': charger_to_vehicle_ratio,
        'average_order_price': average_order_price,
        'average_charging_price': average_charging_price,
        'peak_charging_price': peak_charging_price,
        'average_assign_prob': average_assign_prob
    }

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): The global data dict containing the global instance data with:
    - "fleet_size" (int): Number of EVs in the fleet.
    - "max_time_steps" (int): Maximum number of time steps, e.g., 5 minutes as a time step, 24 hours corresponds to 288 time steps.
    - "total_chargers" (int): Total number of chargers.
    - "max_cap" (int): Max battery capacity in kWh.
    - "consume_rate" (list[float]): Battery consumption rate (as a percentage) per time step for each vehicle in the fleet. Length is fleet_size.
    - "charging_rate" (list[float]): Battery charging rate (as a percentage) per time step for each vehicle in the fleet. Length is fleet_size.
    - "assign_prob" (float): Probability of receiving a ride order when the vehicle is in idle status.
    - "order_price" (list[float]): Payments (in dollars) received per time step when a vehicle is on a ride. The payment is assumed to be the same for all vehicles. The length of the list is max_time_steps.
    - "charging_price" (list[float]): Charging price in dollars per kilowatt-hour ($/kWh) at each time step. The price is assumed to be the same for all vehicles. The length of the list is max_time_steps.
    - "initial_charging_cost" (float): Cost (in dollars) incurred for the first connection to a charger.

        state_data (dict): The state data dict containing the solution state data with:
    - "current_step": The index of current time step. (0 <= current_step < max_time_steps)
    - "ride_lead_time": Ride leading time and length is number of fleet_size.
    - "charging_lead_time": Charging leading time and length is number of fleet_size.
    - "battery_soc": Soc of battery of each feet and length is number of fleet_size.
    - "reward": The total reward for all fleets for this time step.
    - "return": The sum of total reward for all fleets from beginning to current time step.

    Returns:
        dict: The feature of current solution, which can represent the current state with:
            - average_battery_soc (float): Average state of charge of the fleet's batteries.
            - proportion_of_charged_vehicles (float): Proportion of vehicles with high state of charge.
            - average_ride_lead_time (float): Average ride leading time across the fleet.
            - average_charging_lead_time (float): Average charging lead time across the fleet.
            - current_progress (float): Current progress of the solution in terms of time steps.
            - current_reward_rate (float): Reward rate per vehicle for the current time step.
            - overall_return_rate (float): Overall return rate per vehicle from the start to the current step.
    """
    # Calculate the average battery state of charge across the fleet
    average_battery_soc = sum(state_data['battery_soc']) / global_data['fleet_size']
    
    # Calculate the proportion of vehicles with a state of charge greater than 80%
    proportion_of_charged_vehicles = sum(1 for soc in state_data['battery_soc'] if soc > 0.8) / global_data['fleet_size']
    
    # Calculate the average ride lead time across the fleet
    average_ride_lead_time = sum(state_data['ride_lead_time']) / global_data['fleet_size']
    
    # Calculate the average charging lead time across the fleet
    average_charging_lead_time = sum(state_data['charging_lead_time']) / global_data['fleet_size']
    
    # Calculate the progress of the current solution as a fraction of the total time steps
    current_progress = state_data['current_step'] / global_data['max_time_steps']
    
    # Calculate the reward rate per vehicle for the current time step
    current_reward_rate = state_data['reward'] / global_data['fleet_size']
    
    # Calculate the overall return rate per vehicle from the start to the current time step
    overall_return_rate = state_data['return'] / (state_data['current_step'] + 1) / global_data['fleet_size']
    
    return {
        'average_battery_soc': average_battery_soc,
        'proportion_of_charged_vehicles': proportion_of_charged_vehicles,
        'average_ride_lead_time': average_ride_lead_time,
        'average_charging_lead_time': average_charging_lead_time,
        'current_progress': current_progress,
        'current_reward_rate': current_reward_rate,
        'overall_return_rate': overall_return_rate
    }