# Data Folder

This data folder contains instances of the Job Shop Scheduling Problem (JSSP) obtained from the [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/).

## Source

The JSSP instances are sourced from the [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/), which is a collection of datas for a variety of operations research problems. The `jobshop1.txt` and `jobshop2.txt` file within the OR-library contains a number of job shop problem instances, which have been split into individual files for easier access and processing.

## Data Format

- Each data file contains the description of a single JSSP instance with the `.jssp` extension.
- Each instance contains following module:
    - The first line for each instance file is job number(`M`) and operation number(`N`).
    - Then is `M` * 2N matrix. `2N` is the number of jobs to be processed.
    - `M` is the number of jobs to be processed. `2N` represents pairs of entries where the first number in the pair is the machine number and the second number is the processing time required for that operation.
    - The data is organized such that each row corresponds to a job, and each pair within a row corresponds to an operation that needs to be performed on a machine in the specified order. 
- The original file from the OR-library contains multiple instances concatenated together. We have processed this file to split each instance into a separate file for convenience.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation

Beasley, J. E. (n.d.). OR-Library: Distributing Test Problems by Electronic Mail. Journal of the Operational Research Society, 41(11), 1069–1072.

## Access

The original file can be accessed at the following URL:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files

When using this data, please ensure to reference the OR-library and the specific file (`jobshop1`) used for your research or project.

