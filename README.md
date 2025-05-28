# FOCCUS_tools_Shom
Tools developped at Shom for the Tolosa-SW configuration in the Horizon Europe FOCCUS project (https://foccus-project.eu/)

These tools are divided into two parts:
- The extraction of certain processes from the hourly outputs of CMEMS-GLO. This requires placing the GLO data files in NetCDF format in a `<data>` folder.
The program generates an hourly 2D NetCDF file that contains the correction to be applied to the 2D outputs of Tolosa-SW.
- Machine learning calibration. This requires placing the outputs in `.csv` format in a `<data>` folder, with one file per tide gauge site (10-minute time step).
Implemented algorithms include Gradient Boosting (in the `<gb>` folder) and LSTM (along with other tested LSTM variants, in the `<lstm>` folder).
The part concerning the LSTM models is launched via a shell script `run.sh`, in which the necessary parameters for program execution are specified (different types of models, univariate/multivariate, which tuner to use for hyperparameter search, etc.).
The program generates models (in `.keras` format) for all types of models specified as input.

