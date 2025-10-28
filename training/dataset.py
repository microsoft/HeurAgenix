from datasets import load_dataset

def get_custom_dataset(split="train"):
    """
    Loads and processes a custom dataset from a JSON file.
    """
    # The user of this code will need to ensure this path points to their data file.
    data_file_path = "path/to/your/file.json"

    data = load_dataset(
        "json",
        data_files=data_file_path,
    )

    # Assuming the JSON data directly maps to the desired structure.
    # this mapping will create 'prompt' and 'answer' fields for each item.
    processed_data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': x['system']},
            {'role': 'user', 'content': x['instruction']}
        ],
        'answer': x['reward']
    })
    
    # load_dataset with a single file usually creates a 'train' split by default.
    # If your data_file_path points to a directory of files (e.g., train.json, test.json),
    # load_dataset will create splits based on those filenames.
    if split not in processed_data:
        available_splits = list(processed_data.keys())
        raise ValueError(f"Split '{split}' not found in dataset. Available splits: {available_splits}. Please check your data_file_path or the structure of your data.")
        
    return processed_data[split]