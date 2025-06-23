# HeurAgenix
HeurAgenix is a novel framework based on LLM, designed to generate, evolve, evaluate, and select heuristic algorithms for solving combinatorial optimization problems. It leverages the power of large language models to autonomously handle various optimization tasks with minimal human intervention. This framework ensures high adaptability, flexibility, and performance in solving both classical and novel optimization problems.

![Framework Overview](doc/framework.png)

# Prepare

## Set up environment
To set up the environment, run the following command:
```bash
pip install -r requirements.txt
```

## Set up LLM
Currently, the framework supports GPT from Azure using tokens and api based model.

1. Fill in the parameters in json file.
Azure GPT config:
```json
{
    "type": "azure_apt",

    "api_type": "azure",
    "api_base": "...",
    "api_version": "...",
    "azure_endpoint": "...",
    "model": "...",
    "temperature": 0.7,
    "top-p": 0.95,
    "max_tokens": 3200,

    "max_attempts": 50,
    "sleep_time": 10
}
```
API model config:
```json
{
    "type": "api_model",

    "url": "...",
    "api_key": "...",
    "model": "...",
    "temperature": 0.7,
    "top-p": 0.95,
    "max_tokens": 3200,

    "max_attempts": 50,
    "sleep_time": 10
}
```
Local model config:
```json
{
    "type": "local_model",

    "temperature": 0.7,
    "top-p": 0.95,
    "max_tokens": 1600,
    "model_path": "...",

    "max_attempts": 50,
    "sleep_time": 10
}
```
2. Test the LLM activation by:
Modify the `config_file` in chat.py and run
```bash
python chat.py
```

## Prepare Data
### Data for Classical CO Problem
Data sources and formatting requirements for TSP, CVRP, JSSP, MaxCut, and MKP are detailed in the respective readme files.

| Problem                                    | Data Source                                                                                     |
|--------------------------------------------|-------------------------------------------------------------------------|
| Traveling Salesman Problem (TSP)           | [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)    |
| Capacitated Vehicle Routing Problem (CVRP) | [VRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/)         |
| Job Shop Scheduling Problem (JSSP)         | [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/)     |
| Maximum Cut Problem (MaxCut)               | [OPTSICOM](https://grafo.etsii.urjc.es/optsicom/maxcut.html#instances)  |
| Multidimensional Knapsack Problem (MKP)    | [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/) |

### Data For DPOSP
To generate the data for the Dynamic Production Order Scheduling Problem (DPOSP, new in paper):
1. Modify the parameters (production line num, order num, production rate distribution, etc.) in `src/problems/dposp/data/generate_data.py` file.
2. Run the following command to generate data:
```bash
python src/problems/dposp/generate_data.py
```

### Data Structure
It is recommended to organize data into this structure `output/{problem}/data/(train_data, validation_data, test_data, smoke_data)`.
  
- **Evolution Data**: Used by LLM to extract evolution policy during heuristic evolution. Typically consists of small instances either manually designed or sampled from the data.  
- **Validation Data**: Used for evaluating and filtering heuristics during evolution.  
- **Test Data**: Used for testing heuristics or heuristic selection.  
- **Smoke Data**: Used for quick testing of generated or evolved heuristics to check for obvious bugs. Usually consists of small, manually designed instances and includes `previous_operations.txt` (pre-operations for the test) and `smoke_data` (test data).  
  
For example, a common TSP data structure is as follows:  
output/
tsp/
data/
smoke_data/: previous_operations.txt, smoke_data.tsp
test_data/: a280.tsp, bier127.tsp, ...
evolution_data/: case_1.tsp, case_2.tsp, ...
validation_data/: kroA100.tsp, kroA150.tsp, ...

Our built-in data reading interface can handle the standard data format above. If your data is in a new format, you need to override the `load_data` function in `env.py`.  

# Structure and Format
Each problem is independent and share the similar running steps.

## File Structure 
Each problem contains following files

| File | Description | 
| --------------------------------------- | ----------- | 
| heuristics                              | Necessary for [problem solving](#solve-problem). Can be generate by [basic heuristic generation](#generate-basic-heuristic), [heuristic evolution](#evolve-heuristic) or manually written.
| components.py(*)                        | Necessary for all tasks. Provide solution and operations class to support heuristics.
| env.py(*)                               | Necessary for all tasks. Provide Env class to load data, run heuristics, record and the result.
| problem_state.py                        | Necessary for [heuristic evolution](#evolve-heuristic) and [problem solving](#solve-problem). Provide function to extract the problem state. Can be generated by [problem state generation](Generate Problem State) or manually written.
| prompt/problem_description.txt          | Necessary for all tasks. Text-based problem description.
| prompt/problem_state_description.txt(*) | Necessary for [problem state generation](Generate Problem State) by LLM. Description the problem state's format.
| prompt/problem_state.txt                | Necessary for [heuristic evolution](#evolve-heuristic) and [problem solving](#solve-problem). Provide text-based detailed problem state introduction. Can be generated by [problem state generation](#generate-problemtate) or manually written.
| prompt/special_remind.txt               | Optional for [basic heuristic generation](#generate-basic-heuristic), [heuristic evolution](#evolve-heuristic) to avoid some typically error.

The file marked by (*) are basic files that must by provide manually. Others are optional or can be [generated automatically](#run-tasks-on-heuragenix).

## Components
Components contain solution class to record the current(partial) solution and operator to update the solution. Format of components can be found in [components.template.py](doc/components.template.py).

## Env
Env class works as backend to support all tasks in env.py file. Format of env class can be found in [env.template.py](doc/env.template.py).

## Problem State
Problem state is an abstract representation capturing the high-level features of both the problem instance and its current (partial) solution. It is stored in dict and can works as medium between heuristic, env and LLM.  In our framework, 2 files are necessary:
- Problem state extraction function in `problem_state.py` with following 3 functions:
    - def get_instance_problem_state(instance_data: dict) -> dict: Extract instance problem state from instance data.
    - def get_solution_problem_state(instance_data: dict) -> dict: Extract solution problem state from instance data and solution.
    - def get_observation_problem_state(problem_state: dict) -> dict: Extract core problem state as observation.

- Problem state description file in `problem_state.txt`:
    - instance_data: Loaded instance data from load_data function in env.
    - current_solution: Current solution instance.
    - key_item: The key value to evaluate solution and calculated by get_key_value in env. 
    - helper_function: Helper functions to assist heuristics that provided by env.
    - instance_problem_state: extracted by get_instance_problem_state.
    - solution_problem_state: extracted by get_solution_problem_state.

## Heuristic Format
For unified management, all heuristics shared same format:
```python
def heuristic_name(problem_state: dict, algorithm_data: dict, **kwargs) -> tuple[Operator, dict]:
```
problem_state(necessary) contains the problem state.
algorithm_dat(optional) contains the some hyper-parameters for some algorithms
Heuristic returns the target operator that runs on solution to get update with some other information in dict.

Notes: The file name and heuristic_name should keep algin, such as nearest_neighbor_991d.py contains function nearest_neighbor_991d.

# Run Tasks on HeurAgenix
## Generate Problem State 
The problem state can either be written manually or generated automatically by:

```bash
python generate_problem_state.py -p <problem> [-m] [-l <llm_config_file>]
```

Parameters:
- `-p`, `--problem`: Specifies the type of problem for which you want to generate the problem state. Choose from the available problem pool.
- `-m`, `--smoke_test`:  Includes a flag to perform a smoke test, which verifies the validity of the generated problem states through quick checks.
- `-l`, `--llm_config_file`:  Specifies the path to the LLM configuration file to be used. Default is `azure_gpt_4o.json`.

The process will generate the problem state code `problem_state.py` and corresponding description file `problem_state_description.txt` in `output/{problem}/generate_evaluation_function`.

## Generate Basic Heuristic
Basic heuristics serve as seeds for subsequent evolution and can be either manually crafted or automatically generated by:

```bash
python generate_basic_heuristic.py -p <problem> [-m] [-s <source>] [-l <llm_config_file>] [-pp <paper_path>] [-r <related_problems>] [-d <reference_data>]
```

Parameters:
- `-p`, `--problem`: Specifies the type of combinatorial optimization problem.
- `-s`, `--source`: Determines the source of heuristic generation. Options include 'llm' for using a language model, 'paper' for methods from research literature, and 'related_problem' for adapting from similar problems. Default is 'llm'.
- `-l`, `--llm_config_file`: Path to the language model configuration file. Default is `azure_gpt_4o.json`.
- `-m`, `--smoke_test`: Optional flag for performing a quick smoke test to verify the generated heuristics.
- `-pp`, `--paper_path`: Path to a LaTeX paper file or directory containing heuristic methods.
- `-r`, `--related_problems`: List of related problem types to inform heuristic development. Default is 'all'.
- `-d`, `--reference_data`: Optional path to reference datasets, used when generating heuristics tailored to specific data distributions.

The generated heuristics are stored in `output/{problem}/generate_heuristic`. 

## Evolve Heuristic

Heuristic evolution is central to evolve basic heuristic by:

```bash
python evolve_heuristic.py -p <problem> -e <seed_heuristic> [-m] [-l <llm_config_file>] [-ed <evolution_dir>] [-vd <validation_dir>] [-pe <perturbation_heuristic>] [-pr <perturbation_ratio>] [-pt <perturbation_time>] [-i <max_refinement_round>] [-f <filter_num>] [-r <evolution_rounds>] [-d <reference_data>] 
```

Parameters:
- `-p`, `--problem`: Specifies the type of problem for which you want to generate the problem state. Choose from the available problem pool.
- `-e`, `--seed_heuristic`: The initial seed heuristic to be evolved.
- `-ed`, `--evolution_dir`: Directory where the evolution dataset is stored. Default is `evolution_data`.
- `-vd`, `--validation_dir`: Directory where the validation dataset is stored. Default to `validation_data`.
- `-pe`, `--perturbation_heuristic`: Optional name or path for a perturbation heuristic to introduce diverse strategy variations.
- `-pr`, `--perturbation_ratio`: Proportion of solution operations to be altered during perturbation. Default is 0.1.
- `-pt`, `--perturbation_time`: Maximum number of perturbation attempts per evolution cycle. Default is 1000.
- `-i`, `--max_refinement_round`: Number of rounds used in refining heuristics. Default is 5.
- `-f`, `--filter_num`: How many top-performing heuristics to retain following validation. Default is 1.
- `-r`, `--evolution_rounds`: Total number of evolution iterations. Default is 3.
- `-m`, `--smoke_test`:  Includes a flag to perform a smoke test, which verifies the validity of the generated problem states through quick checks.
- `-l`, `--llm_config_file`:  Specifies the path to the LLM configuration file to be used. Default is `azure_gpt_4o.json`.

The evolved heuristics are stored in `output/{problem}/evolution_result/{seed_heuristic}`. 

## Solve Problem

To apply a heuristic or heuristic selector by:

```bash
python launch_hyper_heuristic.py -p <problem> -e <heuristic> [-l <llm_config_file>] [-d <heuristic_dir>] [-t <test_case>] [-n <iterations_scale_factor>] [-m <steps_per_selection>] [-c <num_candidate_heuristics>] [-b <rollout_budget>] [-r <result_dir>]
```

Parameters:
- `-p`, `--problem`: Specifies the type of combinatorial optimization problem to solve (required).
- `-e`, `--heuristic`: Specifies which heuristic function or strategy to apply. Options include:
  - `<heuristic_function_name>`: Directly specify a heuristic function.
  - `'llm_hh'`: Utilizes LLM for rapid heuristic selection from the directory.
  - `'random_hh'`: Randomly selects a heuristic from the directory.
  - `'or_solver'`: Uses an exact OR solver, where applicable.
- `-d`, `--heuristic_dir`: Directory containing heuristics for llm_hh or random_hh. Default is 'basic_heuristics'.
- `-t`, `--test_data`: Name or path of test data or directory for test cases. Defaults to the complete set in `test_data`.
- `-l`, `--llm_config_file`: Path to LLM configuration. Defaults is `azure_gpt_4o.json`.
- `-n`, `--iterations_scale_factor`: Scale factor determining total heuristic steps relative to problem size. Default is 2.0.
- `-m`, `--steps_per_selection`: Number of steps executed per heuristic selection in LLM mode. Default is 5.
- `-c`, `--num_candidate_heuristics`: Number of candidate heuristics considered in LLM mode. 1 represents select by LLM without TTS. Default is 1.
- `-b`, `--rollout_budget`: Number of Monte-Carlo evaluations per heuristic in LLM mode. 0 represents select by LLM without TTS. Default is 0.
- `-r`, `--result_dir`: Target directory for saving results. Default is 'result'.

The solution and evaluation are stored in `output/{problem}/{test_data}/{result}/{seed_heuristic}`. 

## Run on new problems

To setup new problems, please:
- Set up new folder in `src/problems/{problem_name}`
- Implement `env.py` and `components.py` in `src/problems/{problem_name}`
- Provide `problem_description.txt` and `problem_state_description.txt` in `src/problems/{problem_name}/prompt`
- [Set up llm](#set-up-llm) and [Prepare data](#prepare-data) if necessary.
- [Run tasks](#run-tasks-on-heuragenix) as needed.


# Visualize the Project Workflow  
  
To explore the project's workflow visually without executing the actual processes, you can use the Streamlit application. This tool provides an interactive way to understand the framework's components and flow.  
  
To launch the visualization interface, run the following command:  
  
```bash  
streamlit run doc/app.py
```
 
This command will open a web-based interface where you can visually navigate through the project's architecture and processes. Please note that this visualization is for demonstration purposes only and does not perform any actual optimization tasks.

