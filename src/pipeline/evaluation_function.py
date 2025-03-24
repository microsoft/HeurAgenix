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
        dict: The feature of global data, which can represents the global data with:
            - average_consume_rate (float): Average battery consumption rate across the fleet.
            - average_charging_rate (float): Average battery charging rate across the fleet.
            - peak_customer_arrivals (int): Maximum number of customer arrivals in any time step.
            - average_order_price (float): Average order price per minute across all time steps.
            - average_charging_price (float): Average charging price per kWh across all time steps.
            - charger_to_ev_ratio (float): Ratio of total chargers to fleet size.
            - time_steps_per_day (int): Total number of time steps available in a day.
            - charging_session_to_time_resolution_ratio (float): Ratio of charging session time to time resolution.
    """
    # Calculate average consumption rate across the fleet
    average_consume_rate = sum(global_data['consume_rate']) / global_data['fleet_size']

    # Calculate average charging rate across the fleet
    average_charging_rate = sum(global_data['charging_rate']) / global_data['fleet_size']

    # Determine the peak customer arrivals
    peak_customer_arrivals = max(global_data['customer_arrivals'])

    # Calculate the average order price per minute
    average_order_price = sum(global_data['order_price']) / global_data['max_time_steps']

    # Calculate the average charging price per kWh
    average_charging_price = sum(global_data['charging_price']) / global_data['max_time_steps']

    # Calculate the ratio of chargers to EVs
    charger_to_ev_ratio = global_data['total_chargers'] / global_data['fleet_size']

    # Determine the number of time steps per day
    time_steps_per_day = global_data['max_time_steps']

    # Calculate the ratio of charging session time to time resolution
    charging_session_to_time_resolution_ratio = global_data['charging_session_time'] / global_data['time_resolution']

    # Return a dictionary containing all the calculated features
    return {
        'average_consume_rate': average_consume_rate,
        'average_charging_rate': average_charging_rate,
        'peak_customer_arrivals': peak_customer_arrivals,
        'average_order_price': average_order_price,
        'average_charging_price': average_charging_price,
        'charger_to_ev_ratio': charger_to_ev_ratio,
        'time_steps_per_day': time_steps_per_day,
        'charging_session_to_time_resolution_ratio': charging_session_to_time_resolution_ratio
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
        dict: The feature of current solution, which can represents the current state with:
            - average_operational_status (float): Average operational status across the fleet.
            - average_time_to_next_availability (float): Average time to next availability across the fleet.
            - average_battery_soc (float): Average battery state of charge across the fleet.
            - completion_ratio (float): Ratio of current reward to potential maximum reward.
            - progress_ratio (float): Progress in terms of completed time steps.
            - average_reward_per_ev (float): Average reward received per EV at current step.
            - average_return_per_ev (float): Average accumulated return per EV so far.
            - idle_ev_ratio (float): Ratio of idle EVs in the fleet.
            - charging_ev_ratio (float): Ratio of EVs currently charging in the fleet.
    """
    # Calculate average operational status across the fleet
    average_operational_status = np.mean(state_data['operational_status'])

    # Calculate average time to next availability across the fleet
    average_time_to_next_availability = np.mean(state_data['time_to_next_availability'])

    # Calculate average battery state of charge across the fleet
    average_battery_soc = np.mean(state_data['battery_soc'])

    # Calculate completion ratio as ratio of current reward to potential maximum reward
    current_customer_arrivals = global_data['customer_arrivals'][state_data['current_step']]
    completion_ratio = state_data['reward'] / current_customer_arrivals if current_customer_arrivals else 0

    # Calculate progress ratio as progress in terms of completed time steps
    progress_ratio = state_data['current_step'] / global_data['max_time_steps']

    # Calculate average reward received per EV at current step
    average_reward_per_ev = state_data['reward'] / global_data['fleet_size']

    # Calculate average accumulated return per EV so far
    average_return_per_ev = state_data['return'] / global_data['fleet_size']

    # Calculate ratio of idle EVs in the fleet
    idle_ev_ratio = np.sum(state_data['operational_status'] == 0) / global_data['fleet_size']

    # Calculate ratio of EVs currently charging in the fleet
    charging_ev_ratio = np.sum(state_data['operational_status'] == 2) / global_data['fleet_size']

    # Return a dictionary containing all the calculated features
    return {
        'average_operational_status': average_operational_status,
        'average_time_to_next_availability': average_time_to_next_availability,
        'average_battery_soc': average_battery_soc,
        'completion_ratio': completion_ratio,
        'progress_ratio': progress_ratio,
        'average_reward_per_ev': average_reward_per_ev,
        'average_return_per_ev': average_return_per_ev,
        'idle_ev_ratio': idle_ev_ratio,
        'charging_ev_ratio': charging_ev_ratio
    }