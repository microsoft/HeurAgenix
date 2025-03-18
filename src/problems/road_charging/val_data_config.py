import os
import numpy as np
import csv
import json
from collections import defaultdict
from src.problems.road_charging.utils import csv_to_list
from src.problems.road_charging.TripRequests import TripRequests
from src.problems.road_charging.ChargingStations import ChargingStations

def generate_config(
						op_start_hour,
						total_time_steps,
						time_step_minutes,
						total_evs,  # Also used as the number of EVs (N)
						total_chargers,
						committed_charging_block_minutes,
						renewed_charging_block_minutes,
						demand_scaling,
						baseline_prices_fname,
						real_time_prices_fname,
						init_SoCs,
						customer_arrivals_fname,
						per_minute_rates_fname,
						trip_records_fname,
						saved_trips_fname
					):

	config = {
		"total_time_steps": total_time_steps,
		"time_step_minutes": time_step_minutes,
		"total_evs": total_evs,
		"committed_charging_block_minutes": committed_charging_block_minutes,
		"renewed_charging_block_minutes": renewed_charging_block_minutes,
		"operation_start_hour": op_start_hour,
		"demand_scaling": demand_scaling,
		"ev_params": None,
		"trip_params": None,
		"charging_params": None,
		"other_env_params": None,
	}

	# Define charging parameters (using station 1 as the example)
	config["charging_params"] = {
		1: {
			"location": (0, 0),
			"maximum_capacity": total_chargers,
			"available_chargers": total_chargers,
			# "one_time_fee": 15,
			# "occupy_per_min_cost": 0,
			# "queue_per_min_cost": 0,
			"baseline_prices": csv_to_list(baseline_prices_fname),
			"real_time_prices": csv_to_list(real_time_prices_fname),
		}
	}

	# Define EV parameters
	config["ev_params"] = {
		"N": total_evs,
		"SoC_drop_per_min": [0.002] * total_evs,
		"SoC_added_per_min": [0.0166] * total_evs,
		"battery_capacity_kwh": [72] * total_evs,
		"init_SoCs": init_SoCs,
		"min_SoC": 0.1,
		"max_SoC": 1.0,
		"travel_cost": 3, # per minute
		"idle_cost": 3, # per minute.
	}

	# Define trip parameters
	config["trip_params"] = {
		"customer_arrivals_fname": customer_arrivals_fname,
		"per_minute_rates_fname": per_minute_rates_fname,
		"saved_trips_fname": saved_trips_fname,
		"trip_records_fname": trip_records_fname,
		"operation_start_hour": op_start_hour,
		"dt": time_step_minutes,
	}
	
	return config

def save_config(config, filename="config.json"):
	"""
	Save the configuration dictionary to a JSON file.

	Parameters:
		config (dict): The configuration dictionary.
		filename (str): The filename to save the configuration.
	"""
	def convert(o):
		if isinstance(o, tuple):
			return list(o)
		raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
	
	with open(filename, "w") as f:
		json.dump(config, f, default=convert, indent=4)
	print(f"Configuration successfully saved to {filename}")

def rescale_price(data, new_max, new_min):
	old_min, old_max = data.min(), data.max()
	scaled_data = (data - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
	return scaled_data

if __name__ == "__main__":
	# # Generate a configuration for training.
	# # For training, we might leave 'real_time_prices' and 'init_SoCs' empty (or None)
	# # so that the environment can generate random data each episode.
	total_evs = 200
	total_chargers = 10
	resolution = 15
	start_hour = 0
	total_time_steps = int((24-start_hour)*60//resolution)

	price_types = {
		"nyiso":1,
		"caiso":2,
	}
	demand_types = {
		"all_days, demand_scaling = 0.95": 1,
		"all_days, demand_scaling = 0.9": 2,
		"all_days, demand_scaling = 0.8": 3,
		"else": 4,
	}
	SoC_types = {
		"0.75-0.8": 1,
		"0.2-0.8": 2,
		"1.0": 3,
		"0.2": 4
	}
	price_type = 1
	demand_type = 1
	SoC_type = 1
	demand_scaling = {1: 0.95, 2: 0.9, 3: 0.8}.get(demand_type, 1.0)
 
	validation_num = 20

	env_data_path = "env_data"
	input_path = "validation" 
	type_path = f"price{price_type}_demand{demand_type}_SoC{SoC_type}_{total_chargers}for{total_evs}_{start_hour}to24_{resolution}min"
 
	base_price_fname = f"nyiso_rescaled_lmp_30min_2020-06-01.csv"
	if price_type == 2:
		base_price_fname = f"caiso_rescaled_lmp_30min_2023-04-20.csv"
	
	os.makedirs(os.path.join(input_path, type_path), exist_ok=True)
	
	train_config = generate_config(
		op_start_hour = start_hour,
		total_time_steps=total_time_steps,
		time_step_minutes=resolution,
		total_evs=total_evs,
		total_chargers=total_chargers,
		committed_charging_block_minutes=resolution,
		renewed_charging_block_minutes=resolution,
		demand_scaling=demand_scaling,
		baseline_prices_fname=os.path.join(env_data_path, base_price_fname),
		real_time_prices_fname=None, 
		init_SoCs=[],  
		customer_arrivals_fname=os.path.join(env_data_path,f"customer_arrivals_most_trips_{resolution}min.csv"),
		per_minute_rates_fname=os.path.join(env_data_path,f"per_minute_rates_{resolution}min_2019-04.csv"),
		trip_records_fname=os.path.join(env_data_path,"finalize_uber_trips2019-04.csv"),
		saved_trips_fname=None,
	)


	validation_path = os.path.join(input_path, type_path)

	for i in range(1, 1 + validation_num):
		os.makedirs(os.path.join(validation_path, f"instance{i}"), exist_ok=True)
		saved_trips_fname = os.path.join(validation_path, f"instance{i}", f"trip_data{i}.json")
		charging_prices_fname = os.path.join(validation_path, f"instance{i}", f"price_data{i}.csv")

		# ----------------- generate trip data ----------------- 
		trip_requests = TripRequests(train_config["trip_params"])
		trip_requests.reset(np.random)
		# trip_requests.customer_arrivals = [int(np.ceil(x/200*total_evs)) for x in trip_requests.customer_arrivals]
		trip_requests.rescale_customer_arrivals(int(total_evs*demand_scaling))

		for current_timepoint in range(total_time_steps):
			num_requests = int(trip_requests.customer_arrivals[int(start_hour*(60/resolution))+current_timepoint]) 
			trip_requests.sample_requests(num_requests, current_timepoint)
		# print("total trips:", len(trip_requests.trip_queue))

		# Check trip requests before saving
		if not trip_requests.trip_queue:
			raise ValueError(f"Trip requests are empty for instance {i}")

		grouped_durations = defaultdict(list)
		for request in trip_requests.trip_queue.values():
			raised_time = request.get("raised_time")
			trip_duration = request.get("trip_duration")
			trip_fare = request.get("trip_fare")
			grouped_durations[raised_time].append((trip_duration, trip_fare))
		# print(grouped_durations)

		with open(saved_trips_fname, "w") as f:
			json.dump(grouped_durations, f, indent=4)
  
		# ----------------- generate price data ----------------- 
		charging_stations = ChargingStations(train_config["charging_params"])
		charging_stations.reset(np.random)
		charging_stations.update_real_time_prices()
		charging_prices = charging_stations.stations[1]["real_time_prices"]
		with open(charging_prices_fname,"w", newline="") as f:
			writer = csv.writer(f)
			for item in charging_prices:
				writer.writerow([item])  
  
		# ----------------- generate init_SoCs ----------------- 
		if SoC_type == 1:
			init_SoCs = list(np.round(np.random.uniform(0.75, 0.8, size=total_evs), decimals=4))
		if SoC_type == 2:
			init_SoCs = list(np.round(np.random.uniform(0.2, 0.8, size=total_evs), decimals=4))
  		# print("init_SoCs:", init_SoCs)

		validation_config = generate_config(
			op_start_hour=start_hour,
			total_time_steps=total_time_steps,
			time_step_minutes=resolution,
			total_evs=total_evs,
			total_chargers=total_chargers,
			committed_charging_block_minutes=resolution,
			renewed_charging_block_minutes=resolution,
			demand_scaling=demand_scaling,
			baseline_prices_fname=charging_prices_fname,  # No baseline prices needed for validationuation.
			real_time_prices_fname=None,
			init_SoCs=init_SoCs,
			customer_arrivals_fname=os.path.join(env_data_path,f"customer_arrivals_most_trips_{resolution}min.csv"),
			per_minute_rates_fname=os.path.join(env_data_path,f"per_minute_rates_{resolution}min_2019-04.csv"),
			trip_records_fname=os.path.join(env_data_path,"finalize_uber_trips2019-04.csv"),
			saved_trips_fname=saved_trips_fname,
		)

		validation_config_fname = os.path.join(validation_path, f"instance{i}", f"validation_config{i}.json")
		save_config(validation_config, validation_config_fname)

