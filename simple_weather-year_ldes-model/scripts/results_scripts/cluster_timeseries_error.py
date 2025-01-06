import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os
import plotly.graph_objects as go

#functiont that extracts the cluster parameters given a file path
def extract_years_weights(filepath: str, model_type: int):
    
    if model_type == 3:
        model = filepath.split("\\")[-1]

        years = [int(model[14:18]),int(model[19:23]),int(model[24:28])]
        weights = [int(model[29]),int(model[31]),int(model[33])]
        
        output = {
            'years': years,
            'weights': weights
            }
    if model_type == 2:
        model = filepath.split("\\")[-1]

        years = [int(model[21:25]),int(model[26:30])]
        weights = [int(model[39]),int(model[41])]
        
        output = {
            'years': years,
            'weights': weights
            }
    if model_type == 1:
        model = filepath.split("\\")[-1]

        years = [int(model[13:17])]
        weights = [1]
        
        output = {
            'years': years,
            'weights': weights
            }

    return output

# function to compute timeseries error between weighted clusters and reference (full horizon)
def compute_timeseries_error(path: str, model_type: int):

    #extract cluster parameters
    cluster_params = extract_years_weights(path, model_type)

    # load original timeseries
    df_ts_original = pd.read_csv('simple_weather-year_ldes-model/data_tables/time_varying_parameters.csv')
    df_ts_original.columns = ['timesteps','demand','solar_cf','offshore_cf','onshore_cf']
    df_ts_original = df_ts_original[4:]
    # Conversion of value formats
    df_ts_original['timesteps'] = pd.to_datetime(df_ts_original['timesteps'])
    df_ts_original['demand'] = pd.to_numeric(df_ts_original['demand'])
    df_ts_original['solar_cf'] = pd.to_numeric(df_ts_original['solar_cf'])
    df_ts_original['offshore_cf'] = pd.to_numeric(df_ts_original['offshore_cf'])
    df_ts_original['onshore_cf'] = pd.to_numeric(df_ts_original['onshore_cf'])

    df_ts_artificial = pd.DataFrame
    columns_to_apply_weights = ['demand','solar_cf','offshore_cf','onshore_cf']

    for i in range(0,len(cluster_params['years'])):
        #filter the original ts to the given year
        df_temp = df_ts_original[df_ts_original['timesteps'].dt.year == cluster_params['years'][i]]
        #apply the weighting factor
        df_temp.loc[:, columns_to_apply_weights] = df_temp.loc[:, columns_to_apply_weights] * int(cluster_params['weights'][i])
        #assign/append to artificial timeseries
        if df_ts_artificial.empty:
            df_ts_artificial = df_temp
        else:
            df_ts_artificial = pd.concat([df_ts_artificial,df_temp])
    
    #create column to identify hour of year, for grouping, and remove date
    df_ts_artificial['hour_of_year'] = (df_ts_artificial['timesteps'] - pd.to_datetime(df_ts_artificial['timesteps'].dt.year.astype(str) + '-01-01')).dt.total_seconds() // 3600
    df_ts_artificial.drop(columns=['timesteps'], inplace=True)
    df_ts_artificial=(df_ts_artificial.groupby('hour_of_year')).sum()
    #Divide by n_WYs to produce mean
    df_ts_artificial[columns_to_apply_weights] *= 0.1

    #modify original series to group (mean) by hour of year
    df_ts_original['hour_of_year'] = (df_ts_original['timesteps'] - pd.to_datetime(df_ts_original['timesteps'].dt.year.astype(str) + '-01-01')).dt.total_seconds() // 3600
    df_ts_original.drop(columns=['timesteps'], inplace=True)
    df_ts_original=(df_ts_original.groupby('hour_of_year')).mean()

    #compute timeseries errors against reference (original ts)
    df_error = df_ts_artificial.copy()
    df_error[columns_to_apply_weights] = df_error[columns_to_apply_weights] / df_ts_original[columns_to_apply_weights] -1

    # generate error output dictionary including absolute and signed 
    output = {
        'demand_error': {
            'absolute': abs(df_error['demand']).mean(),
            'signed': (df_error['demand']).mean()
        },
        'solar_cf_error': {
            'absolute': abs(df_error['solar_cf']).mean(),
            'signed': (df_error['solar_cf']).mean()
        },
        'offshore_cf_error': {
            'absolute': abs(df_error['offshore_cf']).mean(),
            'signed': (df_error['offshore_cf']).mean()
        },
        'onshore_cf_error': {
            'absolute': abs(df_error['onshore_cf']).mean(),
            'signed': (df_error['onshore_cf']).mean()
        },
        'overall_cf_error': {
            'absolute': (abs(df_error['onshore_cf']).mean()+abs(df_error['offshore_cf']).mean()+abs(df_error['solar_cf']).mean())/3,
            'signed': ((df_error['onshore_cf']).mean()+(df_error['offshore_cf']).mean()+(df_error['solar_cf']).mean())/3
        },
        'compound_error': {
            'absolute': (abs(df_error['onshore_cf']).mean()+abs(df_error['offshore_cf']).mean()+abs(df_error['solar_cf']).mean()+abs(df_error['demand']).mean())/4,
            'signed': ((df_error['onshore_cf']).mean()+(df_error['offshore_cf']).mean()+(df_error['solar_cf']).mean()+(df_error['demand']).mean())/4
        }
    }
    return output

# function that aggregates all results from a directory given a regex query
def results_in_dir(directory, regex_query, model_type):

    output = pd.DataFrame

    # search through the provided directory
    for file in os.listdir(os.fsencode(directory)):
        filename = os.fsdecode(file)
        full_path = directory+"\\"+filename
        # print(full_path)
        if re.match(regex_query,filename):
            #if output is not yet initialised, initialise it
            print('Extracting Results:',filename)
            data = pd.DataFrame(compute_timeseries_error(full_path,model_type))
            cluster_params = extract_years_weights(filename,model_type)
            data['Years'] = ",".join(str(x) for x in cluster_params['years'])
            data['Weights'] = ",".join(str(x) for x in cluster_params['weights'])
            if output.empty:
                output = data
            else:
                output = pd.concat([output,data])

    return output

# debug_test_path = 'model_cluster_2011_2012_2013_1_8_1.netcdf'
# print(pd.DataFrame(compute_timeseries_error(debug_test_path,3)))
result_n_3 = results_in_dir('simple_weather-year_ldes-model/results/n_year_cluster_runs',r"^model_cluster_\d{4}_\d{4}_\d{4}_[a-zA-Z0-9_]*\.netcdf$",3)
print(result_n_3)

result_n_3.to_csv('simple_weather-year_ldes-model/debug/timeseries_error.csv')