# energy-system-modelling-with-LDES
A repository for work related to improving LDES modelling. This README document will be developed further to help users navigate the programe files following initial submission of the conference paper.

General Guidelines are as follows:

All files/code/data relevant to the 'Exploring the imapct of interannual dynamics on long-duration energy storage in energy system models' paper submission can be found within the weather_year_clustering_LDES_study folder. This folder is organise as per the following:

1. case_study_config, is a folder that contains key case study variables as csv tables, such as a list of single, two, and three year clusters.
2. custom_mathematics, hosts custom_mathematics files used by Calliope to enforce non-standard constraints
3. data_tables, hosts permanent and temporary data_tables used by python during model runs. >original_timeseries>time_varying_parameters.csv aggregates the timeseries data for renewable capacity factors and demand sourced from renewables.ninja and ENTSOE respectively.
4. model_config, hosts configuration files for various instances of Calliope models, setup as per standard Calliope best practice. Please see https://calliope.readthedocs.io/en/v0.7.0.dev5/.
5. python, hosts various python scripts used to execute various instances of model as well as plot figures used in the paper itself. Separate scripts exist for the reference, single year, two year, and three year model types. 
6. env_calliope_ldes_study.yaml, is the virtual environment setup file for use with mamba (or conda) to import relevant libraries. It may be that individual systems require additional configuration to run the scripts but this environment file should point users with python experience in the right direction.
7. License, states the license agreement for this code.
