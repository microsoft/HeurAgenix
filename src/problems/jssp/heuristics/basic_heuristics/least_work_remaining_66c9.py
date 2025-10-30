from src.problems.jssp.components import Solution, AdvanceOperator

def least_work_remaining_66c9(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Dispatch rule selecting the job with the smallest remaining workload (sum of operation times from its current operation index to completion). At each decision point it scans all unfinished jobs, computes tail workload, and chooses the argmin, returning an AdvanceOperator to append that job’s next operation to its designated machine queue. This is a best-improvement choice over the candidate set of unfinished jobs; tie-breaking follows first minimum encountered.
    Unique aspects:
    - Operates on the cumulative “tail” time rather than the next-operation time, prioritizing jobs close to completion and aggressively reducing WIP/flow time when remaining times are heterogeneous.
    - Strictly constructive dispatch: only appends the next operation; no reordering of existing machine queues and no consideration of machine availability or queue congestion.
    - Compatible with the one-to-one machine–operation mapping; AdvanceOperator uses job_operation_sequence and job_operation_index to place the next operation.
    Inputs used: job_operation_time, unfinished_jobs, job_operation_index. Others are ignored.
    Time complexity per call: O(sum over unfinished jobs of remaining operations length). Memory: O(1). Myopic behavior may starve long-tail jobs until their remaining time diminishes.


    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "job_operation_time" (numpy.ndarray):  The time cost for each operation in target job.
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
            - "job_operation_index" (list[int]): The index of the next operation to be scheduled for each job.

    Returns:
        AdvanceOperator: Operator to advance the selected job's next operation.
        dict: Empty dictionary as no algorithm data is updated.
    """
    # Extract necessary information from problem_state
    job_operation_time = problem_state["job_operation_time"]
    unfinished_jobs = problem_state["unfinished_jobs"]
    job_operation_index = problem_state["job_operation_index"]

    # Initialize the least work remaining and corresponding job ID
    min_work_remaining = float('inf')
    job_to_advance = None

    # Iterate over unfinished jobs to find the one with the least work remaining
    for job_id in unfinished_jobs:
        remaining_time = sum(job_operation_time[job_id][job_operation_index[job_id]:])
        if remaining_time < min_work_remaining:
            min_work_remaining = remaining_time
            job_to_advance = job_id

    # If no job is found, return None, {}
    if job_to_advance is None:
        return None, {}

    # Create and return the AdvanceOperator for the job with the least work remaining
    return AdvanceOperator(job_to_advance), {}