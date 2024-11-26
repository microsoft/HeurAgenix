import json
import pandas as pd
import numpy as np


# # data_file = "/content/gdrive/MyDrive/RoadCharge/data/"
data_file = "C:\\Users\\zhangling\\OneDrive - Microsoft\\3 Research projects\\2024EV\\codes\\data\\"

config = {
    "fleet_size": 10,
    "total_chargers": 3,
    "step_length": 15, # how many minutes in one time step
    "time_horizon": 60, # how many minutes we are scheduling for
    "max_cap": 72, # EV's battery capacity in kWh
    "connection_fee": 1.5, # fee for connecting to the charging station ($)
    "time_SoCfrom0to100": 60, # Time in minutes to charge the battery fully
    "time_SoCfrom100to0": 480, # Time in minutes to fully discharge the battery
    "assign_prob": 0.5, # probability of getting assigned an order if remaining idle
    "trip_time_fpath": [data_file+'RT_mean_24hrs.csv', data_file+'RT_std_24hrs.csv'], # data to randomly generated passenger trip time
    "trip_fare_fpath" : data_file+'trip_fare_24hrs.csv', # unit fare is calculated per minute, multiply by step length to get fare for a step
    "charging_price_fpath": data_file+"LMP_24hrs.csv", # $
   " data_fpath": data_file,
    "save_path": data_file,
    "additional": { # provide additional information, for example, if we want to sample different charging prices each round
    "all_chargingPrice_fpath": data_file + '2023LMP_hourly.csv', # File path to access all LMP data for 2024
    "Q1_dates_fpath": data_file+"2023Q1_dates.csv", # list of dates in 1st quarter
    "Q2_dates_fpath": data_file+"2023Q2_dates.csv", # list of dates in 2nd quarter
    "Q3_dates_fpath": data_file+"2023Q3_dates.csv", # list of dates in 3rd quarter
    "Q4_dates_fpath": data_file+"2023Q4_dates.csv" # list of dates in 4th quarter
    }
}

with open(data_file+"config.json", "w") as json_file:
    json.dump(config, json_file, indent=4)

print("JSON configuration file created: config.json")

