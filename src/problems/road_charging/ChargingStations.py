import random
import pandas as pd
import numpy as np
import os
from utils import csv_to_list

class ChargingStations:
	def __init__(self, config=None):
		if config is None:
			config = {
				1: {
					"location": (0, 0),
					"maximum_capacity": 5,
					"available_chargers": 5,
					"baseline_prices": [],
					"real_time_prices": [],
				}
			}
		self.stations = config
		self.max_charging_capacity = sum(charger["maximum_capacity"] for charger in config.values())

	def reset(self, rng):
    
		self.rng = rng
  
		# reset available chargers
		for station_id in self.stations.keys():
			self.stations[station_id]["available_chargers"] = self.stations[station_id]["maximum_capacity"]
			self.stations[station_id]["real_time_prices"] = self.stations[station_id]["baseline_prices"]
		# reset open stations
		self.open_stations = list(self.stations.keys())  # List of charger IDs with available slots >= 1


	def get_real_time_prices(self):
     
		first_station = next(iter(self.stations.values()))  # Get the first station

		return first_station["real_time_prices"]


	def reset_cap(self, cap):
     
		for station_id in self.stations.keys():
			self.stations[station_id]["maximum_capacity"] = cap
			self.stations[station_id]["available_chargers"] = cap
		self.max_charging_capacity = sum(charger["maximum_capacity"] for charger in self.stations.values())

	def update_real_time_prices(self):
		for station_id in self.stations.keys():
			baseline_prices = np.array(self.stations[station_id]["baseline_prices"]) 

			# Calculate upper and lower bounds for price variation
			lb = np.max([np.min(baseline_prices) - np.abs(np.min(baseline_prices)) * 0.5, 0])  # Ensure lower bound is non-negative
			ub = np.max(baseline_prices) + np.abs(np.max(baseline_prices)) * 0.5
			
			# Add random noise to baseline prices
			noise = self.rng.normal(0, 0.01, len(baseline_prices))

			# Clip the new prices between lb and ub, then store them
			self.stations[station_id]["real_time_prices"] = np.round(np.clip(baseline_prices + noise, lb, ub), decimals=4).tolist()



	def adjust_occupancy(self, station_id, increment):
		if station_id in self.stations:
			self.stations[station_id]["available_chargers"] = max(min(self.stations[station_id]["available_chargers"]+increment,
						self.stations[station_id]["maximum_capacity"]),0)

	def update_open_stations(self):

		self.open_stations = [
			station_id for station_id, station_info in self.stations.items() if station_info["available_chargers"] >= 1
		]
		
		return self.open_stations
	
	def get_dynamic_resource_level(self):
		
		total_available_chargers = sum(station_info['available_chargers'] for _, station_info in self.stations.items())
		
		return int(total_available_chargers)

