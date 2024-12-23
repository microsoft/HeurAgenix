import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pprint
import math


def config(fleet_size, num_chargers, connection_fee_scaling, data_paths, instance_count, data_type):
    
    env_data_path = data_paths["env"]
    specific_scenario_path = data_paths["scenarios"]
    
    with open(os.path.join(env_data_path, "payment_rates_data.json"), 'r') as json_file:
        payment_rates_data = json.load(json_file) # per minute payment
    
    with open(os.path.join(env_data_path, 'ride_time_distribution_data_rounded.json'), 'r') as json_file:
        ride_time_probs_data = json.load(json_file)
    
    with open(os.path.join(env_data_path, 'order_assign_probs_data.json'), 'r') as json_file:
        order_assign_probs_data = json.load(json_file)
        
    fleet_size = fleet_size          # Number of vehicles in the fleet
    n_chargers = num_chargers          # Number of charging stations available
    time_step_size = 15     # Duration of one time step in minutes
    start_minute = 0        # Simulation start time in minutes
    end_minute = 1440       # Simulation end time in minutes
    # connection_fee = 15    # Fee for connecting to a charging station ($)
    low_SoC = 0.1           # Threshold for low SoC
    max_cap = 72
    minute_SoC100to0 = 480
    minute_SoC0to100 = 60
    
    # Compute derived parameters based on specified parameters
    t_0 = int(start_minute / time_step_size)   # Index of the start time in time steps
    t_T = int(end_minute / time_step_size)     # Index of the end time in time steps
    max_time_step = t_T - t_0                 # Total number of time steps in the simulation
    d_rate = round(1 / minute_SoC100to0 * time_step_size, 3)
    c_rate = round(1 / minute_SoC0to100 * time_step_size, 3)
    c_r = c_rate * max_cap

    ride_data_type = data_type["ride"]
    charging_data_type = data_type["charging"]
    SoC_data_type = data_type["SoC"]
    # payment rates are per minute, convert to per time step
    payment_rates = payment_rates_data[ride_data_type]
    assign_probs = order_assign_probs_data[ride_data_type+f"_{time_step_size}"]
    # charging_prices = pd.read_csv(os.path.join(env_data_path, f"charging_prices_{charging_data_type}.csv"))
    charging_prices = pd.read_csv(os.path.join(env_data_path, f"{charging_data_type}_charging_prices.csv"))["Charging Prices"].tolist()
    initial_SoCs = pd.read_csv(os.path.join(env_data_path, f"{SoC_data_type}_initial_SoCs_{fleet_size}EVs.csv"))["SoCs"].tolist()
    connection_fee = np.max(charging_prices) * max_cap * connection_fee_scaling
    print("charging_prices:", charging_prices)
    # print("initial_SoCs:", initial_SoCs)
    
    # time window adjusted vectors
    w = np.repeat(payment_rates, int(60 / time_step_size))[t_0:t_T] * time_step_size
    p = np.repeat(charging_prices, int(60 / time_step_size))[t_0:t_T]
    rho = np.repeat(assign_probs, int(60 / time_step_size))[t_0:t_T]
    
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
    
    bin_edges = ride_time_probs_data['bin_edges']
    probs = ride_time_probs_data['probabilities'][ride_data_type]
    
    ride_data_instances = []
    for i in range(instance_count):
        ride_data_instance = add_ride_data_instance(fleet_size, time_step_size,
                                                    max_time_step, bin_edges, probs, rho)
        
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
                            bin_edges, probs, order_probs):

    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]

    data_instance = np.zeros((fleet_size, max_time_step))  
    
    for n in range(fleet_size):  
        for step in range(max_time_step):  
            order_prob = order_probs[step]  
            
            if np.random.random() < order_prob:  
                bin_index = np.random.choice(len(bin_centers), size=1, p=probs)[0]  
                x = np.exp(np.random.uniform(low=bin_edges[bin_index], high=bin_edges[bin_index + 1]) ) 
                print('ride time in minutes:', x)
                data_instance[n, step] = math.ceil(x / time_step_size)
                print("ride time in steps:", data_instance[n, step])  
            else:
                data_instance[n, step] = 0  # No order in this time step
        
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
    "scenarios": os.path.join("test_cases")}
    
    data_type = {
    "ride": "all_days",
    "charging": "negative",
    "SoC": "polarized"}
    
    config = config(fleet_size, num_chargers, connection_fee_scaling,
                    data_paths, instance_count, data_type)
    
    X = config["ride_data_instance"]
    print(len(X))
    print(len(X[0]))
    
    plt.step(range(96), X[1])
        
    plt.show()
    
    print(config["order_assign_probs_data"].keys())
    for scenario_name, probs in config["order_assign_probs_data"].items():
        
        if "15" in scenario_name:
            x_values = range(len(probs))  # Time steps (number of bins in the resolution)
            plt.plot(x_values, probs, label=scenario_name)
            
    # Customize the plot
    plt.title(f"Poisson Probabilities for Resolution {10} Minutes", fontsize=14)
    plt.xlabel("Time Steps", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.legend(title="Scenarios", fontsize=10)
    plt.show()
    
    
   
    """
    negative_prices: Charging data includes negative prices
    high_prices: high positive prices
    low_prices: low positive prices
    
    polarized: The SoC is unevenly initialized, with some values very low (close to 0) and others very high (close to 1).
    high_range: SoC values differ slightly but are all within the high range, between 0.8 and 1.
    low_range: SoC values differ slightly but are all within the low range, between 0 and 0.2.
    mid_range: SoC initialized within the middle range, between 0.2 and 0.8.
    same: initial SoC value is the same for all EVs.
    
    all_days: includes rides from all days without differentiating weekdays, weekends, or holidays.
    weekdays:
    weekends:
    holidays:
    nonholidays:
    """
