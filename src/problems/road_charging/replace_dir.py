import os
import json
import glob

# Define the folder containing JSON files
json_folder = "src/problems/road_charging"  # Change this to your actual folder

# Function to convert relative paths to module-style absolute paths
def convert_path(relative_path):
    if relative_path and isinstance(relative_path, str):  # Ensure it's not None
        return relative_path.replace("env_data\\", "src/problems/road_charging/env_data/")

    return relative_path  # Return None or unchanged if not a string

# Get all JSON files in the folder
json_files = glob.glob(os.path.join(json_folder, "*.json"))

# Process each JSON file
for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Check if the key "trip_params" exists and update paths
    if "trip_params" in data:
        for key in ["customer_arrivals_fname", "per_minute_rates_fname", "trip_records_fname"]:
            if key in data["trip_params"]:
                data["trip_params"][key] = convert_path(data["trip_params"][key])

    # Save the updated JSON file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    print(f"Updated {json_file}")

print("✅ All JSON files updated successfully!")
