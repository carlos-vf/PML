# KEEL Module

This folder contains notebooks and experiments specific to datasets in the KEEL format.

## Structure
- `keel_data/`: raw KEEL datasets. Files ending in *tra.dat and *tst.dat are already split in training (tra) and test (tst) (e.g. iris-10-1tra.dat). Numbers are present because of how data was split in the source (https://sci2s.ugr.es/keel/datasets.php). Always match the number that precedes "tra" and "tst" files. 
- `nbooks/`: notebooks performing training/testing on these datasets
- `nbooks_TEMPLATE/`: template notebooks. "not_split_*" vs "split_*" refers to whether it's ALREADY split or not. If "_noisy" is present then there is a cell dedicated to introducing noise, which can be modified.
- `accuracy_scores/`: contains the accuracy scores obtained for each dataset in .csv format

Note: the notebooks can be used for data that comes from other sources, the only necessary thing to modify is how the dataset is imported