Scripts and notebooks in this package should be run with python 3 (they have been tested with python 3.7). The main dependencies are:
- scipy
- numpy
- pandas
- uproot (version 4)
- scikit-learn
- xgboost

## cl3d cluster energy correction and resolution study
### Preprocessing
The preprocessing script `scripts/matching_new.py` takes as input HGCAL TPG ntuples and produces pandas dataframes in HDF files. It is selecting gen particles reaching the HGCAL and matching them with reconstructed clusters.

The bash script launch.sh can be executed to produce the necessary hdf5 files with all necessary arrays from which the plots are made in the next step in notebooks. All the selections are already applied inside matching_new.py script.

### Energy correction and resolution notebook
The dataframes produced at the preprocessing step are used in the notebook `notebooks/jet_calibration.ipynb`. This notebook is performing the following:
- Derive $\eta$ dependent linear energy correction (this is an additive correction) with 200PU electrons
- Produce energy scale and resolution plots, in particular differentially vs  $|\eta|$ and $p_T$
