# Data Folder

This data folder contains instances of the MaxCut problem obtained from the [OPTSICOM](https://grafo.etsii.urjc.es/optsicom/maxcut.html#instances).

## Source

The MaxCut instances are sourced from the [OPTSICOM](https://grafo.etsii.urjc.es/optsicom/maxcut.html#instances) project (set1, set2 and set3), which provides a collection of optimization instances for the MaxCut problem, among others. These instances are used to benchmark algorithms designed to solve MaxCut and have been used in various competitions and research studies.

## Data Format

- Each dataset file contains the description of a single MaxCut problem instance with the `.mc` extension.
- Each instance contains following module:
    - The first line of each instance file contains two integers: `N` and `M`, where `N` is the number of vertices and`M` is the number of edges.
    - The following `M` lines each contain three integers: `u`, `v`, and `w`, representing an edge between vertices `u` and `v` with weight `w`.
    - For edges that do not appear in the file, we assume a default weight of 0.
- It is assumed that the graph is undirected, so each edge is only listed once.
- The original data indexes vertices beginning with 1. In our program, we have adjusted the indices by subtracting 1 from each to start from 0 for consistency. Also output starts from 0 too.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation

When using data from the OPTSICOM project, please acknowledge their contribution by citing their website:

OPTSICOM Project. (n.d.). Optimization for Science and Communication. Universidad Rey Juan Carlos. https://grafo.etsii.urjc.es/optsicom/

## Access

The original data can be accessed at the following URL:
https://grafo.etsii.urjc.es/optsicom/maxcut.html#instances
