# Data Folder 
 
This data folder contains instances of the Traveling Salesman Problem (TSP) obtained from the [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/).
 
## Source 
 
The TSP instances are sourced from the [TSPLIB](http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/), which is a collection of sample instances for the TSP (and related problems) from various sources and of various types. TSPLIB was created by Gerhard Reinelt. 
 
## Data Format

- Each data file contains the description of a single TSP instance with the `.tsp` extension.
- Each instance contains following module:
    - `NAME`: The name of the instance.
    - `COMMENT`: Additional information about the instance, such as its origin or characteristics.
    - `TYPE`: The type of problem, which is TSP for these instances.
    - `DIMENSION`: The number of cities (nodes) in the problem.
    - `EDGE_WEIGHT_TYPE`: The type of edge weights used, which typically denotes how distances between cities are calculated (e.g., Euclidean distance).
    - `NODE_COORD_SECTION`: Each line corresponds to a city and contains the city's index and its coordinates.
- The original data indexes vertices beginning with 1. In our program, we have adjusted the indices by subtracting 1 from each to start from 0 for consistency. Also output starts from 0 too.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation
 
Reinelt, G. (1991). TSPLIB—A traveling salesman problem library. ORSA Journal on Computing, 3(4), 376-384. 

## Access 

The original data can be accessed at the following URL: 
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/
 
