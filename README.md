# HeurAgenix

## Introduction
HeurAgenix is a novel framework based on LLM, designed to generate, evolve, evaluate, and select heuristic algorithms for solving combinatorial optimization problems. It leverages the power of large language models to autonomously handle various optimization tasks with minimal human intervention. This framework ensures high adaptability, flexibility, and performance in solving both classical and novel optimization problems.

![Framework Overview](doc/framework.png)

## Prepare

### Set up environment
To set up the environment, run the following command:
```bash
pip install -r requirements.txt
```

### Set up GPT
Currently, the framework supports activation of GPT using tokens from Azure.

1. Fill in the parameters in `gpt_setting.json`.
2. Test the GPT activation using:
```bash
python chat.py
```

### Prepare Data

Data sources and formatting requirements for TSP, CVRP, JSSP, MaxCut, and MKP are detailed in the respective readme files.

| Problem                             | Data Source                                                                                     | Readme Link                                           |
|-------------------------------------|-------------------------------------------------------------------------------------------------|-------------------------------------------------------|
| Traveling Salesman Problem (TSP)    | [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/)                            | [TSP data readme](src/problems/tsp/data/README.md)    |
| Capacitated Vehicle Routing Problem (CVRP) | [VRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/)                                       | [CVRP data readme](src/problems/cvrp/data/README.md)  |
| Job Shop Scheduling Problem (JSSP)  | [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/)                          | [JSSP data readme](src/problems/jssp/data/README.md)  |
| Maximum Cut Problem (MaxCut)        | [OPTSICOM](https://grafo.etsii.urjc.es/optsicom/maxcut.html#instances)                          | [MaxCut data readme](src/problems/max_cut/data/README.md) |
| Multidimensional Knapsack Problem (MKP) | [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/)                            | [MKP data readme](src/problems/mkp/data/README.md)    |

To generate the data for the Dynamic Production Order Scheduling Problem (DPOSP, new in paper):
1. Modify the parameters (production line num, order num, production rate distribution, etc.) in `src/problems/dposp/data/generate_data.py` file.
2. Run the following command:
    ```bash
    python src/problems/dposp/data/generate_data.py
    ```
3. Refer to the [DPOSP data readme](src/problems/dposp/data/README.md).

## Run on Current Problems (TSP, CVRP, JSSP, MaxCut, MKP, and DPOSP)

### Generate Heuristic
#### Prepare
- **Smoke Data (Optional)**: If you enable the smoke test for the generate heuristic part, you need to provide smoke data in the `src/problems/{problem}/data/smoke_data` folder.
  Smoke data should include two types of files:
    - **Instance Data**: The format should be consistent with other data.
    - **previous_operations.txt (Optional)**: This file records the pre-operations for the smoke test, with each line representing a pre-operation. If not provided, it means there are no pre-operations.

- **Set parameters**: Heuristics can be generated using three methods: from LLM, from paper, and from related problems. You need to set parameters in `generate_heuristic_from_llm/paper/related_problem.py`, including `problem`, `smoke_test_enabled`, `paper_path` (for from paper only), and `related_problems` (for from related problems).

#### Generate
Run the following command to generate heuristics:
```bash
python generate_heuristic_from_llm/paper/related_problem.py
```
The generation process and the generated heuristics will be stored in the `output/{problem}/generate_heuristic` folder.

### Evolve Heuristic
#### Prepare
- **Smoke Data (Optional)**: If you enable the smoke test for the generate heuristic part, you need to provide smoke data in the `src/problems/{problem}/data/smoke_data` folder.
- **Train Data**: Training data should be small-scale instances to enhance the results. You can sample smaller data from the data source or create small data instances yourself and place them in the `src/problems/{problem}/data/train_data` folder.
- **Validation Data (Optional)**: If you enable the validation for the evolution heuristic part, you need to provide validation data. Place the validation data in the `src/problems/{problem}/data/validation_data` folder. If validation is not enabled, this step is not necessary.
- **Set parameters**: Set parameters in `evolution_heuristic.py`, including `problem`, `basic_heuristic`, `perturbation_ratio`, `perturbation_time`, `evolution_round`, `time_limitation`, `smoke_test`, and `validation`.

#### Evolution
Run the following command to evolve heuristics:
```bash
python evolution_heuristic.py
```
The evolution process and the evolved heuristics will be stored in the `output/{problem}/train_result` folder.

### Generate Feature Extractors
#### Prepare
- **Smoke Data (Optional)**: If you enable the smoke test for the generate heuristic part, you need to provide smoke data in the `src/problems/{problem}/data/smoke_data` folder.
- **Set parameters**: Set parameters in `generate_feature_extractor.py`, including `problem`, `smoke_test`.

#### Generate Feature Extractor
Run the following command to generate extractors:
```bash
python generate_feature_extractor.py
```
The generation process and the generated extractor function will be stored in the `output/{problem}/generate_evaluation_function` folder.

### Run Single Heuristic and Heuristic Selector
#### Prepare
- **Test Data**: Provide test data in the `src/problems/{problem}/data/test_data` folder.
- **Set parameters**: Set parameters in `launch_hyper_heuristic.py`, including `problem`, `heuristic_dir`, `validation_for_each_step`, and `hhs`.
    - `validation_for_each_step`: Indicates whether to validate each step, it will make the problem solving slow but to ensure the correctness of heuristic and we can close it when heuristic is stable.
    - `hhs`: List of heuristics to run, where each item represents an experiment:
        - Name of heuristic function: Use the function name of a specific heuristic, such as `nearest_neighbor_f91d`.
        - `random_hh`: Run random heuristic selection.
        - `gpt_hh`: Let LLM select heuristic algorithms.
        - `or_solver`: Run exact OR algorithm (DPOSP only).

- **Feature Extractors and Heuristics**: Pre-generated feature extractors and heuristics are included in the project for convenience, or you can generate them using the steps in ### Generate Feature Extractors and ### Generate Heuristic.

### Run Heuristics
Run the following command to run heuristics:
```bash
python launch_hyper_heuristic.py
```
The running process and the result will be stored in the `output/{problem}/test_result` folder.

## Solving On New Problems
\FrameworkName excels in addressing new problems. When faced with a new problem, the following steps are required:
- Implement an `Env` class in `src/problems/{problem}/env.py` based on [BaseEnv](src/problems/base/env.py) to provide basic support for algorithm execution, including `load_data`, `init_solution`, `get_global_data`, `get_state_data`, and `validation_solution` functionalities.
- Implement a `Solution` class based on [BaseSolution](src/problems/base/component.py) and an `Operation` class based on [BaseOperation](src/problems/base/component.py) in `src/problems/{problem}/component.py` to record solutions and modifications.
- Create the following prompt files in `src/problems/{problem}/prompt`:
    - `problem_description.txt` to describe the new problem.
    - `global_data.txt` to describe the storage format of the instance data for the problem.
    - `state_data.txt` to describe the format of the solution for the problem.
    - `special_remind.txt` (optional) for specific reminders related to the problem during algorithm generation.
- If an OR algorithm is needed as an upper bound, implement the `ORSolver` class in `src/problems/{problem}/or_solver.py`.

## Solve TSP with GLS
We leverage the GLS implementation from [EoH](https://github.com/FeiLiu36/EoH).
- First, generate a solution using our framework. 
- Then, modify the `def nearest_neighbor` function in [gls_evol.py](https://github.com/FeiLiu36/EoH/blob/7a0d7cf73cb3fbcd2cb9c2586e04c6969106fb92/examples/user_tsp_gls/gls/gls_evol.py) in EoH to return our solution as the initial solution. 
- Finally, run the following command in the EoH project to get the results:
```bash
cd examples/user_tsp_gls
python runEoH.py
```


## Future Work
- Currently, some terminology and variable names in the code differ from those in the paper. In the future, we will standardize them according to the paper's descriptions, including:
    - In the code: `global data feature`, `state data feature`, `mathematical analysis`, are referred to in the paper as `instance feature`, `solution feature`, `detailed heuristic design`.
    - Control variables of heuristic algorithms are stored in `algorithm_data` and hyperparameters in `kwargs` in the code, but both are stored in `algorithm_data` in the paper.
- Many parameters are hardcoded in the current implementation. In the future, we will provide a more flexible structure for the processes of generation, evolution, feature extraction, and heuristic utilization.
- We will provide a unified interface to handle various types of data.
- Support for key-based GPT and other LLMs will be added.
