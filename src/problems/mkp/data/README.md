# Data Folder

This data folder contains instances of the Multidimensional Knapsack Problem (MKP) obtained from the [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/).

## Source

The MKP instances are sourced from the [OR-library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/), which is a collection of data sets for a variety of operations research problems. The files `mknap1.txt`, `mknap2.txt`, `mknapcb1.txt` through `mknapcb9.txt` within the OR-library contain a number of multidimensional knapsack problem instances.

## Data Format

- Each data file contains the description of a single MKP instance with the `.mkp` extension.
- `mknap1-{j}.mkp` refers to the j-th case in the file `mknap1.txt`. `mknapcb{i}-{j}.mkp` refers to the j-th case in the file `mknapcb{i}.txt`. Instances not following this pattern are from `mknap2.txt`.
- The first line of each instance file contains two integers: `N` and `M`, where `N` is the number of items and `M` is the number of dimensions (constraints).
- The second line contains `N` integers, representing the values of the items.
- The next `M` lines form an `M * N` matrix, where each row corresponds to a dimension and each column corresponds to an item. The entries in the matrix represent the resource consumption of item for dimension.
- The last line contains `M` integers, representing the resource limits for each dimension.
- The objective is to select a subset of items such that the total value is maximized and the resource consumption for each dimension does not exceed its limit.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation 

When using the data from the OR-library in your research, please cite the following:

Beasley, J. E. (n.d.). OR-Library: Distributing Test Problems by Electronic Mail. Journal of the Operational Research Society, 41(11), 1069–1072.

## Access

The original data can be accessed at the following URL:
https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/