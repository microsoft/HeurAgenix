import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt


def csv_to_list(fname=None):
	if fname is None:
		return []
	else:
		df = pd.read_csv(fname, header=None, dtype=str)  
		
		# Try converting first row to float; if fails, assume it's a header and remove it
		try:
			df.iloc[0].astype(float)  # Check if first row is numeric
			
		except ValueError:
			df = df.iloc[1:]  # Drop first row if it's non-numeric
		
		df = df.astype(float)  # Convert everything to float
		
		data = df.values.tolist()
		
		# Flatten if it's a single column
		if data and len(data[0]) == 1:
			return [round(row[0],4) for row in data]
		
		return data

def load_file(filename):
	"""
	Load a file based on its extension.
	
	If the file is a CSV, return a pandas DataFrame.
	If the file is a JSON, return a dictionary.

	"""
	if filename is None:
		return {}
	# Get the file extension and convert it to lowercase
	_, ext = os.path.splitext(filename)
	ext = ext.lower()

	if ext == '.csv':
		return pd.read_csv(filename)
	elif ext == '.json':
		with open(filename, 'r') as file:
			return json.load(file)
	else:
		raise ValueError(f"Unsupported file format: {ext}")
	
def get_data_type_meanings():
    data_types = {
        "demand_types": {
            "all_days": 1,
            "weekends": 2,
        },
        "SoC_types": {
            "0.75-0.8": 1,
            "1.0": 2,
            "0.2-0.8": 3,
            "0.2": 4,
        }
    }
    
    for category, mappings in data_types.items():
        print(f"{category}:")
        for key, value in mappings.items():
            print(f"  {key} -> {value}")

    return data_types  # Return the dictionary in case it's needed elsewhere



def moving_average(data, window_size):
	return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def visualize_trajectory(agents_trajectory):

	N, T = agents_trajectory["Reward"].shape

	time_steps = range(T + 1)

	plt.figure(figsize=(15, 10))

	for n in range(N):
		plt.subplot(3, 1, 1)
		plt.plot(time_steps, agents_trajectory['SoC'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
		plt.title('Battery State of Charge Over Time')
		plt.xlabel('Time Step')
		plt.ylabel('Battery SoC')
		plt.legend(loc='upper right')

		plt.subplot(3, 1, 2)
		plt.plot(time_steps, agents_trajectory['TimeToNextAvailability'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
		plt.title('Time to Next Availability Over Time')
		plt.xlabel('Time Step')
		plt.ylabel('Time to Availability')
		plt.legend(loc='upper right')

		plt.subplot(3, 1, 3)
		plt.plot(time_steps, agents_trajectory['OperationalStatus'][n, :], marker='o', linestyle='-', label=f'Agent {n}')
		plt.title('Operational Status Over Time')
		plt.xlabel('Time Step')
		plt.ylabel('Status (0=Idle, 1=Serving, 2=Charging)')
		plt.legend(loc='upper right')

	plt.tight_layout()
	plt.show()

def print_ev_rewards_summary(Reward):
	
		total_rewards = Reward.sum(axis=1)
		for ev, reward in enumerate(total_rewards):
			print(f"EV {ev}: Total Reward = {reward}")

		max_ev = total_rewards.argmax()
		min_ev = total_rewards.argmin()
		print(f"EV {max_ev} earned the most with {total_rewards[max_ev]}.")
		print(f"EV {min_ev} earned the least with {total_rewards[min_ev]}.")

def plot_scores(scores, log_scale=False):
	plt.figure(figsize=(10, 5))
	if log_scale:
		log_scores = np.log(scores)
		plt.plot(log_scores, label='Log Scores')
		plt.yscale('log')
		plt.title('Scores and Log Scores Over Episodes')
	else:
		plt.plot(scores, label='Scores')
		plt.title('Scores Over Episodes')
	
	plt.xlabel('Episode')
	plt.ylabel('Score')
	plt.legend()
	plt.grid(True)
	plt.show()
	


# 6 6:15
# 0 1

# 0/(60/15) = 0 ... 0
# 1/(60/15) = 0 ... 1
# 2/(60/15) = 0 ... 2
# 3/(60/15) = 0 ... 3
# 4/(60/15) = 1 ... 0
# to_price_idx: current_timepoint/(60 // 15)



