import json
import pandas as pd
import numpy as np


def config_env(data_path:str, save_path:str):
    # Input the path for your data file
    config = {
        "fleet_size": 5,
        "total_chargers": 2,
        "step_length": 15, # how many minutes in one time step
        "time_horizon": 1440, # how many minutes we are scheduling for
        "max_cap": 72, # EV's battery capacity in kWh
        "connection_fee": 1.5, # fee for connecting to the charging station ($)
        "time_SoCfrom0to100": 60, # Time in minutes to charge the battery fully
        "time_SoCfrom100to0": 480, # Time in minutes to fully discharge the battery
        "prob_fpath": data_path+"assign_prob.csv", # probability of getting assigned an order if remaining idle
        "trip_fare_fpath" : data_path+'trip_fare_24hrs.csv', # unit fare is calculated per minute, multiply by step length to get fare for a step
        "charging_price_fpath": data_path+"LMP_24hrs.csv", # $
        "data_fpath": data_path,
        "save_path": save_path,
        "additional": { # provide additional information, for example, if we want to sample different charging prices each round
        "all_chargingPrice_fpath": data_path + '2023LMP_hourly.csv', # File path to access all LMP data for 2024
        "Q1_dates_fpath": data_path+"2023Q1_dates.csv", # list of dates in 1st quarter
        "Q2_dates_fpath": data_path+"2023Q2_dates.csv", # list of dates in 2nd quarter
        "Q3_dates_fpath": data_path+"2023Q3_dates.csv", # list of dates in 3rd quarter
        "Q4_dates_fpath": data_path+"2023Q4_dates.csv" # list of dates in 4th quarter
        }
    }
    
    with open(data_path+"config.json", "w") as json_file:
        json.dump(config, json_file, indent=4)
    
    print("JSON configuration file created: config.json")

    return config

    
if __name__ == "__main__":

    data_path = "C:\\Users\\zhangling\\OneDrive - Microsoft\\3 Research projects\\2024EV\\codes\\csv_data\\"
    save_path = "C:\\Users\\zhangling\\OneDrive - Microsoft\\3 Research projects\\2024EV\\codes\\results\\"
    config = config_env(data_path, save_path)

    time_bin_width = config["step_length"] # 10 minutes as a step
    max_time_steps = int(24 * 60 / time_bin_width)
    time_steps = list(range(0, max_time_steps))

    assign_prob = pd.read_csv(config['prob_fpath']).iloc[:, 0].tolist()
    rho = np.repeat(assign_prob, int(60 / time_bin_width)) 

    if time_bin_width == 10:
        ride_time_buckets = [10, 20, 30, 40, 50, 60, 70, 80, 90]
        with open(data_path+"ride_time_pmf_10min.json", "r") as f:
            ride_time_probs = json.load(f)

    elif time_bin_width == 15:
        ride_time_buckets = [15, 30, 45, 60, 75, 90]
        with open(data_path+"ride_time_pmf_15min.json", "r") as f:
            ride_time_probs = json.load(f)

    ride_time_bins = [int(item/time_bin_width) for item in ride_time_buckets]  # Discretized ride times

    config["ride_time_bins"] = ride_time_bins
    config["ride_time_probs"] = ride_time_probs

    max_cases = 2

    for case in range(max_cases):
        case_samples = []
        for i in range(config["fleet_size"]):
            sample = []
            for step in time_steps:
                if np.random.random() < rho[step]:
                    random_ride_time = random_ride_time = np.random.choice([rt for rt in ride_time_bins if rt > 0],
                                                            p=[prob for prob in ride_time_probs if prob>0] )
                    sample.append(int(random_ride_time))
                else:
                    sample.append(0)

            case_samples.append(sample)

        print(len(case_samples), len(case_samples[0]))

        config["ride_time_samples"] = case_samples

        # Save each case as a JSON file
        with open(data_path+f"case_{case + 1}.json", "w") as json_file:
            json.dump(config, json_file, indent=4)

        
