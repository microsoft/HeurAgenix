# This file is generated generate_evaluation_function.py and to renew the function, run "python generate_evaluation_function.py"

import numpy as np

def get_global_data_feature(global_data: dict) -> dict:
    """Extract features from global data for the JSSP.

    Args:
        global_data (dict): The global data dict containing:
            - "job_operation_sequence" (numpy.ndarray): Each job's sequence of operations.
            - "job_operation_time" (numpy.ndarray): Time cost for each operation of each job.
            - "job_num" (int): Total number of jobs.
            - "machine_num" (int): Total number of machines.

    Returns:
        dict: A dictionary containing calculated features from the global data.
    """
    job_operation_sequence = global_data["job_operation_sequence"]
    job_operation_time = global_data["job_operation_time"]
    job_num = global_data["job_num"]
    machine_num = global_data["machine_num"]

    # Calculate features
    average_operation_time = np.mean(job_operation_time)
    max_operation_time = np.max(job_operation_time)
    min_operation_time = np.min(job_operation_time)
    std_deviation_operation_time = np.std(job_operation_time)
    job_operation_time_range = np.ptp(job_operation_time)
    average_job_length = np.mean([len(job) for job in job_operation_sequence])
    max_job_length = np.max([len(job) for job in job_operation_sequence])
    min_job_length = np.min([len(job) for job in job_operation_sequence])
    machine_utilization = np.count_nonzero(job_operation_sequence) / (job_num * machine_num)
    job_diversity = len(np.unique(job_operation_sequence))

    # Compile features into a dictionary
    features = {
        "average_operation_time": average_operation_time,
        "max_operation_time": max_operation_time,
        "min_operation_time": min_operation_time,
        "std_deviation_operation_time": std_deviation_operation_time,
        "job_operation_time_range": job_operation_time_range,
        "average_job_length": average_job_length,
        "max_job_length": max_job_length,
        "min_job_length": min_job_length,
        "machine_utilization": machine_utilization,
        "job_diversity": job_diversity
    }

    return features

import numpy as np

def get_state_data_feature(global_data: dict, state_data: dict) -> dict:
    """Extract features from state data for the JSSP.
    
    Args:
        global_data (dict): Contains the global static information data.
        state_data (dict): Contains the current dynamic state data.
        
    Returns:
        dict: A dictionary containing calculated features from the state data.
    """
    finished_jobs = state_data["finished_jobs"]
    unfinished_jobs = state_data["unfinished_jobs"]
    job_last_operation_end_times = state_data["job_last_operation_end_times"]
    machine_last_operation_end_times = state_data["machine_last_operation_end_times"]
    current_makespan = state_data["current_makespan"]
    machine_num = global_data["machine_num"]
    job_num = global_data["job_num"]
    
    # Calculate features
    num_finished_jobs = len(finished_jobs)
    num_unfinished_jobs = len(unfinished_jobs)
    average_job_completion = np.mean(job_last_operation_end_times)
    max_job_completion_time = np.max(job_last_operation_end_times)
    min_job_completion_time = np.min(job_last_operation_end_times)
    std_dev_job_completion_time = np.std(job_last_operation_end_times)
    average_machine_completion = np.mean(machine_last_operation_end_times)
    max_machine_completion_time = np.max(machine_last_operation_end_times)
    min_machine_completion_time = np.min(machine_last_operation_end_times)
    std_dev_machine_completion_time = np.std(machine_last_operation_end_times)
    average_idle_time_per_machine = (current_makespan * machine_num - np.sum(machine_last_operation_end_times)) / machine_num
    proportion_of_finished_jobs = num_finished_jobs / job_num
    proportion_of_unfinished_jobs = num_unfinished_jobs / job_num
    
    # Compile features into a dictionary
    features = {
        "num_finished_jobs": num_finished_jobs,
        "num_unfinished_jobs": num_unfinished_jobs,
        "average_job_completion": average_job_completion,
        "max_job_completion_time": max_job_completion_time,
        "min_job_completion_time": min_job_completion_time,
        "std_dev_job_completion_time": std_dev_job_completion_time,
        "average_machine_completion": average_machine_completion,
        "max_machine_completion_time": max_machine_completion_time,
        "min_machine_completion_time": min_machine_completion_time,
        "std_dev_machine_completion_time": std_dev_machine_completion_time,
        "current_makespan": current_makespan,
        "average_idle_time_per_machine": average_idle_time_per_machine,
        "proportion_of_finished_jobs": proportion_of_finished_jobs,
        "proportion_of_unfinished_jobs": proportion_of_unfinished_jobs
    }
    
    return features