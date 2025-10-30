from src.problems.jssp.components import Solution, AdvanceOperator

def first_come_first_served_6c4f(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[AdvanceOperator, dict]:
    """
    Deterministic FCFS job-level scheduler. Each call selects the earliest-arrived unfinished job (the head of unfinished_jobs) and advances exactly its next operation via AdvanceOperator, which appends the job to its required machine’s queue according to job_operation_sequence and increments job_operation_index. No evaluation of processing times, machine states, or alternative jobs; no scanning or tie-breaking beyond the given arrival order. This is a purely constructive, one-step progression policy that enforces arrival-order fairness and leverages the one-to-one machine–operation mapping implicitly through AdvanceOperator. Time complexity per decision: O(1); constant memory. Deterministic and reproducible given a fixed unfinished_jobs order; feasibility and resource checks are delegated to the validation layer. Best suited for baseline/initialization or online arrival settings where arrival order is the governing priority.

    Args:
        problem_state (dict): The dictionary contains the problem state. In this algorithm, the following items are necessary:
            - "unfinished_jobs" (list[int]): List of all unfinished jobs.
        algorithm_data (dict): Not used in this heuristic.
        **kwargs: Additional hyperparameters (not used in this heuristic).

    Returns:
        AdvanceOperator: The operator that advances the first unfinished job.
        dict: Empty dictionary as the algorithm data is not updated.
    """
    
    # Check if there are any unfinished jobs. If not, return None.
    if not problem_state["unfinished_jobs"]:
        return None, {}
    
    # Retrieve the first job from the list of unfinished jobs.
    job_id = problem_state["unfinished_jobs"][0]
    
    # Create an AdvanceOperator to schedule the next operation for the first job.
    operator = AdvanceOperator(job_id=job_id)
    
    # No algorithm data is updated, so we return an empty dictionary.
    return operator, {}