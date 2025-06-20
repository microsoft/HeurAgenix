import os
import numpy as np
import pandas as pd
from src.problems.road_charging.utils import csv_to_list, load_file


class TripRequests:
	def __init__(self, config=None):
		self.customer_arrivals = config.get("customer_arrivals")
		self.per_minute_rates = config.get("per_minute_rates")
		self.saved_data = load_file(config.get("saved_trips_fname"))
		self.trip_records = load_file(config.get("trip_records_fname"))
		self.start_hour = config.get("operation_start_hour", 6)
		self.dt = config.get("dt", 15)

	def reset(self, rng=np.random):
		self.trip_queue = {}
		self.open_requests = []
		self.max_id = 0
  
		self.rng = rng
  
	def update_customer_arrivals(self, std=0.05):
     
		data = np.array(self.customer_arrivals)
		noise_std = std * data  # 5% of each element as std deviation
		simulated_data = self.rng.normal(loc=data, scale=noise_std)

		simulated_data = np.clip(simulated_data, a_min=0, a_max=None)
		simulated_data = np.ceil(simulated_data).astype(int)
		self.customer_arrivals = data
  
	# def rescale_customer_arrivals(self, target):
	# 	# Find the maximum value in customer_arrivals
	# 	max_arrival = max(self.customer_arrivals)

	# 	# Rescale the values so that the largest value becomes self.N
	# 	self.customer_arrivals = [
	# 		int(np.ceil(x / max_arrival * target)) for x in self.customer_arrivals]


	def load_saved_trip_data(self):
		if not self.saved_data:
			raise Exception("self.saved_data is empty.")
		
		self.saved_data = {int(k): v for k, v in self.saved_data.items()}
		pickup_location = (0, 0)
		for raised_time, trips in self.saved_data.items():
			for start_time, duration in trips:
				self.create_request(raised_time, pickup_location, start_time, duration)

	def create_request(self, raised_time, pickup_location, trip_duration, trip_fare):
		self.max_id += 1
		self.trip_queue[self.max_id] = {
			"raised_time": raised_time,
			"pickup_location": pickup_location,
			"trip_duration": trip_duration,
			"status": "open",  # Can be "open" or "completed"
			"trip_fare": trip_fare,
		}
		self.open_requests.append(self.max_id)

	def complete_request(self, req_id):
		if req_id in self.trip_queue and req_id in self.open_requests:
			self.trip_queue[req_id]["status"] = "completed"
			self.open_requests.remove(req_id)
		else:
			raise Exception("Request does not exist.")
	
	def update_open_requests(self, current_timepoint):
		
		self.open_requests = [key for key, value in self.trip_queue.items() 
					 if value['raised_time'] == current_timepoint and value['status'] == 'open']
		return self.open_requests

	def sample_requests(self, num_requests, raised_time):

		if num_requests == 0:
			return None

		if self.per_minute_rates is not None:
			pay_rate = self.per_minute_rates[raised_time]

		sampled_requests = None
		if self.trip_records is not None:
			filtered_records = self.filter(raised_time)
			# sampled_requests = filtered_records.sample(n=num_requests).to_dict('records')
			sampled_indices = self.rng.choice(filtered_records.index, size=num_requests, replace=False)
			sampled_requests = filtered_records.loc[sampled_indices].to_dict('records')

			if sampled_requests is not None:
				for v in sampled_requests:
					raised_time = raised_time
					pickup_location = (0, 0)
					trip_duration = int(round(v['trip_time']))
					trip_fare = round(max(pay_rate * trip_duration, 8),4)
					self.create_request(raised_time, pickup_location, trip_duration, trip_fare)

	def filter(self, current_timepoint):
		if self.trip_records is None:
			return pd.DataFrame()

		self.trip_records['request_datetime'] = pd.to_datetime(self.trip_records['request_datetime'])

		total_minutes = self.start_hour * 60 + current_timepoint * self.dt
		hour = (total_minutes // 60)
		
		filtered_data = self.trip_records[self.trip_records['request_datetime'].dt.hour == hour]
		return filtered_data

	def count_trip_requests(self):
		open_count = 0
		completed_count = 0

		# Iterate through the trip queue dictionary
		for trip in self.trip_queue.values():
			if trip["status"] == "open":
				open_count += 1
			elif trip["status"] == "completed":
				completed_count += 1
		
		return open_count, completed_count
