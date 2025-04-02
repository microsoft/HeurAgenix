# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

def get_global_data_feature(global_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): A dictionary containing global instance data with the following keys:
        - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
        - "max_time_steps" (int): The maximum number of time steps. For example, with a 15-minute time step, 24 hours would correspond to 96 time steps.
        - "time_resolution" (int): The length of a single time step in minutes.  
        - "charging_session_time" (int): The duration of a charging session, defaulting to `time_resolution`.
        - "total_chargers" (int): The maximum number of available chargers.
        - "max_cap" (int): The maximum battery capacity in kilowatt-hours (kWh).
        - "consume_rate" (list[float]): A list representing the battery consumption rate (as a percentage) per time step for each vehicle in the fleet. The list has a length equal to the fleet size.
        - "charging_rate" (list[float]): A list representing the battery charging rate (as a percentage) per time step for each vehicle in the fleet. The list has a length equal to the fleet size.
        - "min_SoC" (float): The safety battery SoC threshold. If an EV's SoC falls below this level, it will no longer be considered for order dispatch, even if available.
        - "max_SoC" (float): The maximum allowable SoC during charging, defaulting to 1.0.
        - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step. The length of the list is equal to max_time_steps.
        - "order_price" (list[float]): A list representing the payment (in dollars) received per minute when a vehicle is on a ride. The payment is assumed to be the same for all vehicles. The length of the list is equal to the number of time steps (max_time_steps).
        - "charging_price" (list[float]): A list representing the charging price (in dollars per kilowatt-hour, $/kWh) at each time step.  The list has a length equal to the max time steps.

    Returns:
        The feature of global data, which can represents the global data with:
            - feature_name (type of value): description
            - feature_name (type of value): description
            ...
    """
    # Calculate average customer arrivals
    average_customer_arrivals = sum(global_data["customer_arrivals"]) / global_data["max_time_steps"]
    
    # Calculate peak customer arrivals
    peak_customer_arrivals = max(global_data["customer_arrivals"])
    
    # Calculate average order price
    average_order_price = sum(global_data["order_price"]) / global_data["max_time_steps"]
    
    # Calculate average charging price
    average_charging_price = sum(global_data["charging_price"]) / global_data["max_time_steps"]
    
    # Calculate fleet to charger ratio
    fleet_to_charger_ratio = global_data["fleet_size"] / global_data["total_chargers"]
    
    # Calculate average consume rate
    average_consume_rate = sum(global_data["consume_rate"]) / global_data["fleet_size"]
    
    # Calculate average charging rate
    average_charging_rate = sum(global_data["charging_rate"]) / global_data["fleet_size"]
    
    # Calculate battery capacity utilization
    battery_capacity_utilization = global_data["max_cap"] * global_data["fleet_size"]
    
    # Extract min and max SoC thresholds
    min_SoC_threshold = global_data["min_SoC"]
    max_SoC_threshold = global_data["max_SoC"]
    
    # Return all calculated features in a dictionary
    return {
        "average_customer_arrivals": average_customer_arrivals,
        "peak_customer_arrivals": peak_customer_arrivals,
        "average_order_price": average_order_price,
        "average_charging_price": average_charging_price,
        "fleet_to_charger_ratio": fleet_to_charger_ratio,
        "average_consume_rate": average_consume_rate,
        "average_charging_rate": average_charging_rate,
        "battery_capacity_utilization": battery_capacity_utilization[0],
        "min_SoC_threshold": min_SoC_threshold,
        "max_SoC_threshold": max_SoC_threshold
    }

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """ Feature to extract the feature of global data.

    Args:
        global_data (dict): A dictionary containing global instance data with the following keys:
        - "fleet_size" (int): The number of electric vehicles (EVs) in the fleet.
        - "max_time_steps" (int): The maximum number of time steps. For example, with a 15-minute time step, 24 hours would correspond to 96 time steps.
        - "time_resolution" (int): The length of a single time step in minutes.  
        - "charging_session_time" (int): The duration of a charging session, defaulting to `time_resolution`.
        - "total_chargers" (int): The maximum number of available chargers.
        - "max_cap" (int): The maximum battery capacity in kilowatt-hours (kWh).
        - "consume_rate" (list[float]): A list representing the battery consumption rate (as a percentage) per time step for each vehicle in the fleet. The list has a length equal to the fleet size.
        - "charging_rate" (list[float]): A list representing the battery charging rate (as a percentage) per time step for each vehicle in the fleet. The list has a length equal to the fleet size.
        - "min_SoC" (float): The safety battery SoC threshold. If an EV's SoC falls below this level, it will no longer be considered for order dispatch, even if available.
        - "max_SoC" (float): The maximum allowable SoC during charging, defaulting to 1.0.
        - "customer_arrivals" (list[int]): A list representing the number of customer arrivals at each time step. The length of the list is equal to max_time_steps.
        - "order_price" (list[float]): A list representing the payment (in dollars) received per minute when a vehicle is on a ride. The payment is assumed to be the same for all vehicles. The length of the list is equal to the number of time steps (max_time_steps).
        - "charging_price" (list[float]): A list representing the charging price (in dollars per kilowatt-hour, $/kWh) at each time step.  The list has a length equal to the max time steps.

        state_data (dict): A dictionary containing the solution state data with the following keys:
        - "current_solution" (Solution): The current action trajectory (solution) for evs.
        - "current_step": The index of the current time step, where (0 <= current_step < max_time_steps).
        - "operational_status": A 1D array of size as fleet_size, where:
          - 0 represents idle,
          - 1 represents serving a trip, and
          - 2 represents charging.
        - "time_to_next_availability": A 1D array of size as fleet_size, indicating the lead time until the fleet becomes available (after completing a trip or finishing charging).
        - "battery_soc": A 1D array of size as fleet_size, representing the battery state of charge in percentage.
        - "reward": The total reward for the entire fleet at the current time step.
        - "return": The accumulated reward for the entire fleet from time step 0 to the current time step.

    Returns:
        The feature of current solution, which can represents the current state with:
            - feature_name (type of value): description
            - feature_name (type of value): description
            ...
    """
    # Calculate average battery SoC
    average_battery_soc = np.mean(state_data["battery_soc"])
    
    # Count idle vehicles
    idle_vehicles = np.sum(state_data["operational_status"] == 0)
    
    # Count serving vehicles
    serving_vehicles = np.sum(state_data["operational_status"] == 1)
    
    # Count charging vehicles
    charging_vehicles = np.sum(state_data["operational_status"] == 2)
    
    # Calculate average time to availability
    average_time_to_availability = np.mean(state_data["time_to_next_availability"])
    
    # Extract current reward
    current_reward = state_data["reward"]
    
    # Extract accumulated return
    accumulated_return = state_data["return"]
    
    # Calculate progress ratio
    progress_ratio = state_data["current_step"] / global_data["max_time_steps"]
    
    # Count vehicles below minimum SoC threshold
    vehicles_below_min_SoC = np.sum(state_data["battery_soc"] < global_data["min_SoC"])
    
    # Count vehicles at maximum SoC threshold
    vehicles_at_max_SoC = np.sum(state_data["battery_soc"] >= global_data["max_SoC"])
    
    # Return all calculated features in a dictionary
    return {
        "average_battery_soc": average_battery_soc,
        "idle_vehicles": idle_vehicles,
        "serving_vehicles": serving_vehicles,
        "charging_vehicles": charging_vehicles,
        "average_time_to_availability": average_time_to_availability,
        "current_reward": current_reward,
        "accumulated_return": accumulated_return,
        "progress_ratio": progress_ratio,
        "vehicles_below_min_SoC": vehicles_below_min_SoC,
        "vehicles_at_max_SoC": vehicles_at_max_SoC
    }