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

def compute_error(df_original,df_artificial,parameter: str):

    dc_original = (df_original[parameter].sort_values(ascending=False).values).astype(np.float32)
    dc_artificial = (df_artificial[parameter].sort_values(ascending=False).values).astype(np.float32)

    abs_deviation = np.divide(abs(dc_original-dc_artificial), dc_original, out=np.zeros_like(abs(dc_original-dc_artificial)), where=dc_original!=0)

    return abs_deviation.mean()

# duration curve error function
def compute_duration_curve_error(path: str, model_type: int):

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

    # filter timerseries to 2010-2019 inclusive
    df_ts_original=df_ts_original[df_ts_original['timesteps'].dt.year >= 2010]
    # remove leap days due to series length mismatches
    df_ts_original=df_ts_original[~((df_ts_original['timesteps'].dt.month == 2) &(df_ts_original['timesteps'].dt.day == 29))]


    df_ts_artificial = pd.DataFrame
    columns_to_apply_weights = ['demand','solar_cf','offshore_cf','onshore_cf']

    for i in range(0,len(cluster_params['years'])):
        #filter the original ts to the given year
        df_temp = df_ts_original[df_ts_original['timesteps'].dt.year == cluster_params['years'][i]]

        #assign/append to artificial timeseries
        if df_ts_artificial.empty:
            df_ts_artificial = df_temp
        else:
            df_ts_artificial = pd.concat([df_ts_artificial,df_temp])

        if cluster_params['weights'][i] > 1:
            # duplicate the timeseries for that year based on its weighting. i.e. for a year weighted as 8, we get that year's timeseries 8 times.
            for j in range(1,(cluster_params['weights'][i]+1)-1):
                df_ts_artificial = pd.concat([df_ts_artificial,df_temp])

    # compute duration curve and duration curve error for each parameter
    demand_error = compute_error(df_ts_original,df_ts_artificial,'demand')
    solar_error = compute_error(df_ts_original,df_ts_artificial,'solar_cf')
    offshore_error = compute_error(df_ts_original,df_ts_artificial,'offshore_cf')
    onshore_error = compute_error(df_ts_original,df_ts_artificial,'onshore_cf')
    combined_absolute_error = (demand_error+solar_error+offshore_error+onshore_error)/4

    return  {
        'demand_error': demand_error,
        'solar_cf_error': solar_error,
        'offshore_cf_error': offshore_error,
        'onshore_cf_error': onshore_error,
        'compound_error': combined_absolute_error
    }

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
            data = pd.DataFrame(compute_duration_curve_error(full_path,model_type),index=[0])
            cluster_params = extract_years_weights(filename,model_type)
            # data['Years'] = ",".join(str(x) for x in cluster_params['years'])
            # data['Weights'] = ",".join(str(x) for x in cluster_params['weights'])
            data['Run'] = f"[{",".join(str(x) for x in cluster_params['years'])}]_[{":".join(str(x) for x in cluster_params['weights'])}]"
            if output.empty:
                output = data
            else:
                output = pd.concat([output,data])

    return output

# debug_test_path = 'model_cluster_2011_2012_2013_1_8_1.netcdf'
# print(pd.DataFrame(compute_timeseries_error(debug_test_path,3)))
# result_n_3 = results_in_dir('simple_weather-year_ldes-model/results/three_year_cluster_runs',r"^model_cluster_\d{4}_\d{4}_\d{4}_[a-zA-Z0-9_]*\.netcdf$",3)
# print(result_n_3)

# result_n_3.to_csv('simple_weather-year_ldes-model/export/timeseries_duration_curve_error.csv')

def is_sequential(number_string: str):
    # Convert string to array of numbers
    numbers = list(map(int, number_string.split(',')))
    diffs = [abs(numbers[0]-numbers[1]),abs(numbers[1]-numbers[2]),abs(numbers[2]-numbers[0])]
    if diffs.count(1) != 2:
        return False
    return True
    # check these are consecutive
    # return all(numbers[i]+1 == numbers[i+1] for i in range(len(numbers)-1))

def timeseries_error_vs_ESOM_error():

    df_timeseries_error = pd.read_csv('simple_weather-year_ldes-model/export/timeseries_duration_curve_error.csv')
    df_cluster_error = pd.read_csv('simple_weather-year_ldes-model/export/data_cluster_box_plots.csv')
    df_timeseries_error = df_timeseries_error.drop_duplicates(subset=['Run'])
    df_cluster_error = df_cluster_error.drop_duplicates(subset=['Run'])


    # Join Results into single PD
    df_timeseries_error.set_index('Run')
    df_cluster_error.set_index('Run')
    df_combined = pd.merge(df_timeseries_error, df_cluster_error, how="left", on=["Run"])

    df_combined = df_combined[['Run','compound_error','Cost Error','Mean Abs Capacity Mix Error','Abs Compound Error']]
    df_combined['Years'] = df_combined['Run'].str[1:15]
    df_combined['Check'] = df_combined['Years'].apply(is_sequential)

    df_combined_sequential =df_combined[ df_combined['Check'] == True]
    df_combined_nonsequential =df_combined[ df_combined['Check'] == False]
    print(df_combined['Years'])
    print(df_combined[df_combined['Run']=='[2017,2015,2016]_[8:1:1]'])

    WY1_colour = 'rgb(102, 153, 255)'
    WY2_colour = 'rgb(51, 204, 204)'
    WY3_colour = 'rgb(51, 51, 153)'

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        name='All Weather Years Contiguous',
        y=df_combined_sequential['Abs Compound Error'],
        x=df_combined_sequential['compound_error'],
        text=df_combined_sequential['Run'],
        mode='markers',
        marker=dict(
            color=WY1_colour,
            size=20,
            line=dict(
                color=WY1_colour,
                width=2
            ))
    ))

    fig.add_trace(go.Scatter(
        name='Non-Contiguous Weather Years',
        y=df_combined_nonsequential['Abs Compound Error'],
        x=df_combined_nonsequential['compound_error'],
        text=df_combined_nonsequential['Run'],
        mode='markers',
        marker=dict(
            color='white',
            size=20,
            line=dict(
                color=WY1_colour,
                width=2
            )
    )))

    fig.update_yaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    # tick0=-0.05, 
    # dtick=0.05
    )
    fig.update_xaxes(
    ticks='outside',
    showline=True,
    linecolor='black',
    gridcolor='rgba(0, 0, 0, 0.2)',
    # tick0=0, 
    # dtick=0.05
    )
    fig.update_layout(
        plot_bgcolor='rgba(255, 255, 255, 0)',
        title=dict(text=f"Preoptimisation Duration Curve Error versus Postoptimisation Absolute Compound Error"),
        yaxis = dict(
            title = dict(
                text='Postoptimisation (Absolute Compound) Error',
                font=dict(size=16)
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        xaxis = dict(
            title = dict(
                text='Preoptimisation (Duration Curve) Error',
                font=dict(size=16)
            ),
            zerolinecolor = 'rgba(0, 0, 0, 0.4)',
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor = 'rgba(255,255,255,0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        # boxmode='group',
    yaxis_tickformat=".02%",
    xaxis_tickformat=".02%",
    )

    fig.write_html('simple_weather-year_ldes-model/export/scatter_timseries_vs_ESOM_error.html', auto_open=True)


timeseries_error_vs_ESOM_error()

