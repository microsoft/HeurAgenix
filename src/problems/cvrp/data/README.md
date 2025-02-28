# Data Folder

This data folder contains instances of the Capacitated Vehicle Routing Problem (CVRP) obtained from the [VRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/).

## Source

The VRP instances are sourced from the [VRPLIB](http://vrp.galgos.inf.puc-rio.br/index.php/en/), which is a library of VRP instances compiled from various sources and includes different types of VRP problems. The VRPLIB was initiated by Ivan Lima and the authors of the reference (Uchoa et al. 2017), and has been maintained by several individuals over the years.

## Data Format

- Each data file contains the description of a single CVRP instance with the `.vrp` extension.
- Each instance contains following module:
    - `NAME`: The name of the instance.
    - `COMMENT`: Additional information such as the number of trucks and the optimal value if known.
    - `TYPE`: The type of problem, which is CVRP for these instances.
    - `DIMENSION`: The number of nodes, including the depot.
    - `EDGE_WEIGHT_TYPE`: The method used to calculate distances between nodes.
    - `CAPACITY`: The maximum load that each vehicle can carry.
    - `NODE_COORD_SECTION`: Each line corresponds to a city and contains the city's index and its coordinates.
- To avoid ambiguity, we add a line VEHICLE: N at the end of the file to indicate that the number of vehicles is N.
- The original data indexes vertices beginning with 1. In our program, we have adjusted the indices by subtracting 1 from each to start from 0 for consistency. Also output starts from 0 too.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation

When using the data from VRPLIB in your research, please cite the following paper:

Uchoa, E., Pecin, D., Pessoa, A., Poggi, M., Vidal, T., & Subramanian, A. (2017). New benchmark instances for the capacitated vehicle routing problem. European Journal of Operational Research, 257(3), 845-858.

Additionally, please acknowledge the creators and maintainers of the VRPLIB:

- Ivan Lima (Creator, 2014)
- Daniel Oliveira (Maintainer, 2016-2018)
- Eduardo Queiroga (Maintainer, 2019-present)

## Access

The original data can be accessed at the following URL:
http://vrp.galgos.inf.puc-rio.br/index.php/en/
