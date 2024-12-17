import json
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt


def config(data_path, save_path, rt_instances, rt_scenario="default"):
    
    with open(data_path + 'payment_rates_data.json', 'r') as json_file:
        payment_rates_data = json.load(json_file) # per minute payment
    
    with open(data_path + 'ride_time_probs_data.json', 'r') as json_file:
        ride_time_probs_data = json.load(json_file)
    
    with open(data_path + 'order_assign_probs_data.json', 'r') as json_file:
        order_assign_data = json.load(json_file)
        
    # print("payment_rates_data:", payment_rates_data)
    # print("order_assign_probs_data:", order_assign_data)
        
    charging_price = pd.read_csv(data_path+"default_charging_price_24hrs.csv").iloc[:, 0].tolist()
    
    # Specify parameters
    fleet_size = 5          # Number of vehicles in the fleet
    n_chargers = 2          # Number of charging stations available
    time_step_size = 15     # Duration of one time step in minutes
    start_minute = 0        # Simulation start time in minutes
    end_minute = 1440       # Simulation end time in minutes
    connection_fee = 1.5    # Fee for connecting to a charging station ($)
    low_SoC = 0.1           # Threshold for low SoC
    max_cap = 72
    minute_SoC100to0 = 480
    minute_SoC0to100 = 60
    
    # Compute derived parameters based on specified parameters
    t_0 = int(start_minute / time_step_size)   # Index of the start time in time steps
    t_T = int(end_minute / time_step_size)     # Index of the end time in time steps
    max_time_steps = t_T - t_0                 # Total number of time steps in the simulation
    d_rate = round(1 / minute_SoC100to0 * time_step_size, 3)
    c_rate = round(1 / minute_SoC0to100 * time_step_size, 3)
    c_r = c_rate * max_cap

    initial_SoC = []
    for _ in range(fleet_size):
        initial_SoC.append(round(np.random.uniform(0, 1),3))
    
    payment_rates = payment_rates_data[rt_scenario]
    rt_bin_edges = ride_time_probs_data['bin_edges']
    rt_probs = ride_time_probs_data['probabilities'][rt_scenario]
    assign_probs = order_assign_data[rt_scenario+f"_{time_step_size}"]
    
    # print("payment_rates:", payment_rates)
    # print("rt_bin_edges:", rt_bin_edges)
    # print("rt_probs:", rt_probs)
    # print("assign_probs:", assign_probs)
    
    # time window adjusted vectors
    w = np.repeat(payment_rates, int(60 / time_step_size))[t_0:t_T] * time_step_size
    p = np.repeat(charging_price, int(60 / time_step_size))[t_0:t_T]
    rho = np.repeat(assign_probs, int(60 / time_step_size))[t_0:t_T]
    
    # write config
    config = {
        "fleet_size":fleet_size,
        "n_chargers": n_chargers,
        "time_step_size": time_step_size,
        "t_0": t_0,
        "t_T": t_T,
        "max_time_steps": max_time_steps,
        "connection_fee($)": connection_fee,
        "low_SoC": low_SoC,
        "initial_SoC": initial_SoC,
        "max_cap": max_cap,
        "d_rate(%)": [d_rate] * fleet_size,
        "c_rate(%)": [c_rate] * fleet_size,
        "c_r(kWh)": [c_r] * fleet_size,
        "payment_rates_data($)": payment_rates_data,
        "order_assign_data": order_assign_data,
        "ride_time_probs_data": ride_time_probs_data,
        "charging_price($/kWh)": charging_price,
        "rt_scenario": rt_scenario,
        "charging_scenario": "default",
        "initial_SoC_scenario": "default",
        "rt_bin_edges": rt_bin_edges,
        "w": w.tolist(),
        "rho": rho.tolist(),
        "p": p.tolist(),
        "data_path": data_path,
        "save_path": save_path,
        }
    
    # save config file
    with open(save_path + "default_config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
        
    generate_rt_data(config, rt_instances, rt_scenario)
        
    return config


def generate_rt_data(config, n_instances, rt_scenario):
    
    fleet_size = config["fleet_size"]
    time_step_size = config["time_step_size"]
    max_time_steps = config["max_time_steps"]
    rt_bin_edges = config["rt_bin_edges"]
    rt_probs = config["ride_time_probs_data"]['probabilities'][rt_scenario]
    
    if config["rt_scenario"] != rt_scenario:
        t_0 = config["t_0"]
        t_T = config["t_T"]
        payment_rates = config["payment_rates_data($)"][rt_scenario]
        assign_probs = config["order_assign_data"][rt_scenario+f"_{time_step_size}"]
        w = np.repeat(payment_rates, int(60 / time_step_size))[t_0:t_T] * time_step_size
        rho = np.repeat(assign_probs, int(60 / time_step_size))[t_0:t_T]

        config["rt_scenario"] = rt_scenario
        config["w"] = w.tolist()
        config["rho"] = rho.tolist()
        
    else:
        rho = config["rho"]
    
    # Generate rt data samples
    bin_centers = [(rt_bin_edges[i] + rt_bin_edges[i + 1]) / 2 for i in range(len(rt_bin_edges) - 1)]
    # print("len(bin_centers):", len(bin_centers))

    for i in range(n_instances):  # each instance contains a matrix of size (fleet_size, max_time_steps)
        random_data_instance = np.zeros((fleet_size, max_time_steps))  # Initialize the matrix for this instance
        
        for n in range(fleet_size):  # Loop through each fleet instance
            for step in range(max_time_steps):  # Loop through each time step
                prob_order = rho[step]  # Probability of receiving an order at this time step
                
                # Generate a random number and compare with prob_order to decide if an order is received
                if np.random.random() < prob_order:  # If order is received
                    bin_index = np.random.choice(len(bin_centers), size=1, p=rt_probs)  # Choose a bin index
                    bin_index = bin_index[0]  # Flatten the array since np.random.choice returns an array
                    rt_minutes = np.random.uniform(low=rt_bin_edges[bin_index], high=rt_bin_edges[bin_index + 1])  # Generate random ride time
                    
                    # Exponentiate and discretize the ride time to steps
                    rt_steps = int(np.exp(rt_minutes) / time_step_size)  
                    random_data_instance[n, step] = rt_steps  # Store the ride time steps for this fleet instance at this time step
                    
                else:
                    random_data_instance[n, step] = 0  # No order in this time step
                    
        # Save the generated data for this instance to a file
        np.save(save_path + f"rt_scenario_{rt_scenario}_instance_{i+1}.npy", random_data_instance)
    
        config["ride_time_instance"] = random_data_instance.tolist()

        # Save modified config file
        with open(save_path + f"config_{rt_scenario}_instance_{i+1}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
        
        plt.figure()
        for i, vehicle_data in enumerate(random_data_instance):
            plt.plot(range(max_time_steps), vehicle_data, marker='o', label=f"Vehicle {i+1}")
            
        plt.show()
        
    return config


def change_charging_price(config, n_instances, data_path, save_path,
                           scenario_name="Q2"):

    time_step_size = config["time_step_size"]
    t_0 = config["t_0"]
    t_T = config["t_T"]
    
    assert scenario_name in ["Q1", "Q2", "Q3", "Q4"]
    
    df = pd.read_csv(data_path+"price_year2023_hourly.csv")
    dates = pd.read_csv(data_path+f"2023{scenario_name}_dates.csv")

    for i in range(n_instances):
        random_date = np.random.choice(dates)

        charging_price = df[df['Local Date']==random_date]['SP-15 LMP'].to_numpy().tolist()
        p = np.repeat(charging_price, int(60 / time_step_size))[t_0:t_T]  # Charging price per time step

        config["charging_price($/kWh)"] = charging_price
        config["p"] = p
        
        with open(save_path + f"/price_scenario_{scenario_name}_instance{i + 1}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)
            
    return config

            
def change_initial_SoC(config, n_instances, save_path,
                        scenario_name="General"):
    
    fleet_size = config["fleet_size"]
    
    if scenario_name == "default":
    
        for i in range(n_instances):
            initial_SoC = []
            for _ in range(fleet_size):
                initial_SoC.append(round(np.random.uniform(0, 1),3))
                
    elif scenario_name == "high":
        for i in range(n_instances):
            initial_SoC = []
            for _ in range(fleet_size):
                initial_SoC.append(round(np.random.uniform(0.8, 1),3))
                
    elif scenario_name == "low":
        for i in range(n_instances):
            initial_SoC = []
            for _ in range(fleet_size):
                initial_SoC.append(round(np.random.uniform(0., 0.2),3))
                
    config["initial_SoC"] = initial_SoC
    
    with open(save_path + f"/initial_SoC_scenario_{scenario_name}_instance{i + 1}.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
        
        
    return config
        

if __name__ == "__main__":

    data_path = "C://Users//zhangling//OneDrive - Microsoft//6 GitHub//on-road-charging//raw_data//"
    save_path = "C://Users//zhangling//OneDrive - Microsoft//6 GitHub//on-road-charging//configuration//"
    
    # Stage 1: Focus on default scenario
    config = config(data_path, save_path, 2, "default")
   
    
