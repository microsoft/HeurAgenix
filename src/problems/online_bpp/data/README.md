# Data Folder

This data folder contains instances of the Bin Packing Problem (BPP) obtained from the [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/) and the [Weibull 5k](https://www.researchgate.net/publication/235709548_Weibull-Based_Benchmarks_for_Bin_Packing) dataset.

## Source

- The files `text_0.txt` to `text_4.txt` are extracted from the [Weibull 5k](https://www.researchgate.net/publication/235709548_Weibull-Based_Benchmarks_for_Bin_Packing) dataset.  
- The files `u500_00.txt` to `u500_19.txt` are extracted from `binpack3.txt` in the [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/).

## Data Format

- Each data file contains the description of a single bpp instance with the `.bpp` extension.
- The first line of each instance file contains two integers: `N` and `M`, where `N` is the capacity of each bin and `M` is the number of items.
- The next `M` lines represents the volume for each item.

## Dataset

- The train_data is a compressed sample from the source, curated to facilitate heuristic evolution.
- The validation_data is a direct sample from the source, used to refine and filter the evolved heuristics.
- The test_data is a direct sample from the source, intended to evaluate the performance of the heuristics.
- The smoke_data dir is used for a quick check to see if the heuristic can run, where smoke_data in the dir is the test intance, and previous_actions are the precursor actions.

## Citation  

When using the data in your research, please cite the following:  

> Beasley, J. E. (1990). OR-Library: Distributing Test Problems by Electronic Mail. *Journal of the Operational Research Society, 41*(11), 1069–1072.  
> DOI: [10.1057/jors.1990.166](https://doi.org/10.1057/jors.1990.166)  

When using the data from the **Weibull 5k** in your research, please cite the following:  

> Castineiras, I., de Cauwer, M., & O'Sullivan, B. (2012). Weibull-Based Benchmarks for Bin Packing. *Integration of AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems (CPAIOR 2012)*, 207–222.  
> DOI: [10.1007/978-3-642-33558-7_17](https://doi.org/10.1007/978-3-642-33558-7_17)  

## Access  

The datasets can be accessed at the following URLs:  
- [OR-Library](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/)  
- [Weibull 5k](https://www.researchgate.net/publication/235709548_Weibull-Based_Benchmarks_for_Bin_Packing)
