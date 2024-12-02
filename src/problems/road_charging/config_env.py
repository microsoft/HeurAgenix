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
        "ride_time_distribution_name": "log-normal",
        # "assign_prob": 0.5, # probability of getting assigned an order if remaining idle
        "prob_fpath": data_path+"assign_prob.csv", # probability of getting assigned an order if remaining idle
        "trip_time_fpath": [data_path+'RT_mean_24hrs.csv', data_path+'RT_std_24hrs.csv'], # data to randomly generated passenger trip time
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

    
if __name__ == "__main__":

    # data_path = "/content/gdrive/MyDrive/RoadCharge/data/"
    data_path = r"data/"
    save_path = r""
    config_env(data_path, save_path)

