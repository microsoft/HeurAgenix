import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
import math


def config(fleet_size, num_chargers, connection_fee_scaling, data_paths, instance_count, data_type):
    """
    Charging Data Type:
    negative_prices: Charging data includes negative prices
    high_prices: high positive prices
    low_prices: low positive prices

    SoC Data Type:
    polarized: The SoC is unevenly initialized, with some values very low (close to 0) and others very high (close to 1).
    high: SoC values differ slightly but are all within the high range, between 0.8 and 1.
    low: SoC values differ slightly but are all within the low range, between 0 and 0.2.
    mid: SoC initialized within the middle range, between 0.2 and 0.8.
    sameHigh/sameLow: initial SoC value is the same for all EVs.

    Ride Data Type:
    all_days: includes rides from all days without differentiating weekdays, weekends, or holidays.
    weekdays
    weekends
    holidays
    nonholidays
    """
    
    env_data_path = data_paths["env"]
    specific_scenario_path = data_paths["scenarios"]
    
    with open(os.path.join(env_data_path, "payment_rates_data.json"), 'r') as json_file:
        payment_rates_data = json.load(json_file) # per minute payment
    
    with open(os.path.join(env_data_path, 'ride_time_distribution_data_updated.json'), 'r') as json_file:
        ride_time_probs_data = json.load(json_file)
    
    with open(os.path.join(env_data_path, 'order_assign_probs_data_updated.json'), 'r') as json_file:
        order_assign_probs_data = json.load(json_file)
        
    fleet_size = fleet_size          # Total number of electric vehicles (EVs) in the fleet
    n_chargers = num_chargers       # Number of charging stations available for the fleet
    time_step_size = 15             # Duration of each simulation step in minutes
    start_minute = 0                # Starting time for the simulation (in minutes)
    end_minute = 1440               # End time for the simulation (in minutes, e.g., 1440 minutes = 24 hours)
    low_SoC = 0.1                   # Threshold for low state of charge (SoC), e.g., 10%
    max_cap = 72                     # Maximum battery capacity of each EV (in kWh)
    minute_SoC100to0 = 480          # Time (in minutes) for an EV to discharge from 100% to 0% SoC
    minute_SoC0to100 = 60           # Time (in minutes) for an EV to charge from 0% to 100% SoC
    
    # Derived parameters based on the specified settings
    t_0 = int(start_minute / time_step_size)   # Index for the start time step
    t_T = int(end_minute / time_step_size)     # Index for the end time step
    max_time_step = t_T - t_0                  # Total number of time steps in the simulation
    d_rate = round(1 / minute_SoC100to0 * time_step_size, 3)  # Discharge rate per time step
    c_rate = round(1 / minute_SoC0to100 * time_step_size, 3)   # Charge rate per time step
    c_r = c_rate * max_cap                     # Charging rate per time step (in kWh)
    
    # Data type mappings
    ride_data_type = data_type["ride"]        # Data type for ride-related information
    charging_data_type = data_type["charging"] # Data type for charging-related information
    SoC_data_type = data_type["SoC"]          # Data type for SoC-related information
    
    # Load external data
    payment_rates = payment_rates_data[ride_data_type]       # Payment rates for the ride data type
    assign_probs = order_assign_probs_data[ride_data_type + f"_{time_step_size}"]  # Order assignment probabilities
    charging_prices = pd.read_csv(os.path.join(env_data_path, f"{charging_data_type}_charging_prices.csv"))["Charging Prices"].tolist()  # Charging prices (per kWh)
    initial_SoCs = pd.read_csv(os.path.join(env_data_path, f"{SoC_data_type}_initial_SoCs_{fleet_size}EVs.csv"))["SoCs"].tolist()  # Initial SoCs for each vehicle
    
    # Connection fee based on maximum charging price
    connection_fee = np.max(charging_prices) * max_cap * connection_fee_scaling  # Connection fee based on the highest charging price and scaling factor
    
    # Adjusted vectors for time window
    w = np.repeat(payment_rates, int(60 / time_step_size))[t_0:t_T] * time_step_size  # Convert payment rates (per minute) to per time step
    p = np.repeat(charging_prices, int(60 / time_step_size))[t_0:t_T]  # Convert hourly charging prices to per time step
    rho = np.repeat(assign_probs, int(60 / time_step_size))[t_0:t_T]  # Convert hourly order assignment probabilities to per time step

    
    # write config
    config = {
        "fleet_size":fleet_size,
        "n_chargers": n_chargers,
        "time_step_size": time_step_size,
        "t_0": t_0,
        "t_T": t_T,
        "max_time_step": max_time_step,
        "connection_fee($)": connection_fee,
        "low_SoC": low_SoC,
        "initial_SoCs": initial_SoCs,
        "max_cap": max_cap,
        "d_rates(%)": [d_rate] * fleet_size,
        "c_rates(%)": [c_rate] * fleet_size,
        "charging_powers(kWh)": [c_r] * fleet_size,
        "payment_rates_data($)": payment_rates_data,
        "order_assign_probs_data": order_assign_probs_data,
        "ride_time_probs_data": ride_time_probs_data,
        "charging_prices($/kWh)": charging_prices,
        "ride_data_type": ride_data_type,
        "charging_data_type": charging_data_type,
        "SoC_data_type": SoC_data_type,
        "w": w.tolist(),
        "rho": rho.tolist(),
        "p": p.tolist()
        }
    
    ride_time_probs_data = pd.DataFrame(ride_time_probs_data)
    ride_scenario_probs = ride_time_probs_data[ride_data_type].values
    
    ride_data_instances = []
    for i in range(instance_count):
        ride_data_instance = add_ride_data_instance(fleet_size, time_step_size,
                                                    max_time_step, ride_time_probs_data, ride_scenario_probs, rho)
        
        config["ride_data_instance"] = ride_data_instance.tolist()
        ride_data_instances.append(ride_data_instance.tolist())
        
        scenario_dir = os.path.join(
        specific_scenario_path,
        f'{ride_data_type}_{charging_data_type}Prices_{SoC_data_type}InitSoC_{n_chargers}for{fleet_size}'
        )
        os.makedirs(scenario_dir, exist_ok=True)
        
        config_file_path = os.path.join(
            scenario_dir, 
            f'config{i+1}_{fleet_size}EVs_{n_chargers}chargers.json'
        )
        with open(config_file_path, 'w') as json_file:
            json.dump(config, json_file)
        
    return config


def add_ride_data_instance(fleet_size, time_step_size, max_time_step,
                            ride_time_probs_data, ride_scenario_probs, order_probs):

    data_instance = np.zeros((fleet_size, max_time_step))  
    
    for n in range(fleet_size):  
        for step in range(max_time_step):  
            order_prob = order_probs[step]  
            
            if np.random.random() < order_prob:  
                row_index = np.random.choice(ride_time_probs_data.index, size=1, p=ride_scenario_probs)
                
                bin_range = ride_time_probs_data.loc[row_index, 'Ride Time Range (Minutes)'].iloc[0]  # Get the range as a string
                lower_bound, upper_bound = map(int, bin_range.split(' - '))  # Split and convert to integers
                
                x = np.random.uniform(lower_bound, upper_bound)
                data_instance[n, step] = int(math.ceil(x / time_step_size))
                
            else:
                data_instance[n, step] = 0  # No order in this time step
    print("data_instance:", data_instance)
        
    return data_instance


# Run example
if __name__ == "__main__":

    fleet_size = 5
    num_chargers = 1
    data_paths = {}
    instance_count = 20
    connection_fee_scaling = 5

    data_paths = {
    "env": os.path.join("env_data"),
    "scenarios": os.path.join("test_cases_adjusted")}
    
    data_type = {
    "ride": "all_days",
    "charging": "negative",
    "SoC": "high"}
    
    config = config(fleet_size, num_chargers, connection_fee_scaling,
                    data_paths, instance_count, data_type)
    
    
    
    
   
   
