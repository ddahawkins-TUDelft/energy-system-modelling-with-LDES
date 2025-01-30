import datetime
import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.express as px
import re
import os

# calliope.set_log_verbosity("info", include_solver_output=True)


#plan
# Load original timeseries
# generate new timeseries .csv for cluster's weather years
#    - probably need to rewrite dates as fake dates so calliope can accept non-consecutive WYs
#    - need to track the mapping function to convert back later
# function for assigning weights to weather years


anchor_weather_year = 2010
appended_weather_year = 2011
weight_anchor_weather_year = 0.1  #TODO: Make decisions around these weights
weight_appended_weather_year = 0.9 #TODO: Make decisions around these weights

def hours_in_year(year):
    """
    Determines the number of hours in a given year.
    
    Parameters:
    year (int): The input year
    
    Returns:
    int: Number of hours in the year
    """
    # Check if the year is a leap year
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
    # Calculate hours
    hours = 366 * 24 if is_leap else 365 * 24
    
    return hours

#resamples the anchor and appended weather year timeseries to new consecutive timeseries beginning 01/01/2010
def generate_timeseries(dict_input):

    #load the original timeseries data
    df_original_timeseries = pd.read_csv(
        "simple_weather-year_ldes-model/data_tables/time_varying_parameters.csv",
        names=["timesteps","var1","var2","var3","var4"]
        )

    #retain structural pointers for calliope to be re-added later
    df_headers = df_original_timeseries.head(5)

    #isolate timesteps content for manipulation
    df_timesteps_values = df_original_timeseries.drop([0,1,2,3,4])

    df_clustered_timeseries_data = pd.DataFrame

    for wy in dict_input['weather_years']:
        df_timeseries_slice = df_timesteps_values[(df_timesteps_values['timesteps']<f"{wy+1}-01-01")&(df_timesteps_values['timesteps']>=f"{wy}-01-01")]
        #if the first in the set, initialise the df and populate with first's values
        if wy == dict_input['weather_years'][0]:
            df_clustered_timeseries_data = df_timeseries_slice
        else:
            df_clustered_timeseries_data = pd.concat([df_clustered_timeseries_data,df_timeseries_slice])
    
    start_ts = pd.Timestamp('2010-01-01')  #arbitrary start date for clusters
    periods = int(df_clustered_timeseries_data.count()['timesteps']) #calculate the length of timeseries to be resampled
    dti = pd.date_range(start=start_ts, periods=periods, freq='h') #generate new timeseries. Method accoutns for leap years
    df_clustered_timeseries_data['timesteps'] = dti

    #time format
    df_clustered_timeseries_data['timesteps']=df_clustered_timeseries_data['timesteps'].dt.strftime("%Y/%m/%d %H:%M")

    df_result = pd.concat([df_headers,df_clustered_timeseries_data])
    df_result.reset_index(drop=True, inplace=True)

    df_result.to_csv(
        path_or_buf=f"simple_weather-year_ldes-model/data_tables/n_cluster_weather_years/cluster_timeseries_{'_'.join(str(x) for x in dict_input['weather_years'])}.csv", 
        index=False, 
        header=False,
        )
    
    return df_result # for debug only, not correct format for proper use
    
#generates timestep weights given anchor/appended weather years and associated weights
def generate_timestep_weights(dict_input):

    #generate timeseries
    generate_timeseries(dict_input)
    #generate dti with appropraite number of timepoints and weights 
    periods = 0
    start_ts = pd.Timestamp('2010-01-01')  #arbitrary start date for clusters
    for wy in dict_input['weather_years']:
        periods = periods +int(hours_in_year(wy))
        if wy == dict_input['weather_years'][0]:
            weights_array = np.full(hours_in_year(wy), dict_input['weights'][dict_input['weather_years'].index(wy)])
        else:
            weights_array = np.concatenate([weights_array,np.full(hours_in_year(wy), dict_input['weights'][dict_input['weather_years'].index(wy)])])
            
    dti = pd.date_range(start=start_ts, periods=periods, freq='h',name='timesteps') #generate new timeseries. Method accoutns for leap years
    
    #assign to export df
    df_result = dti.to_frame().drop(columns=['timesteps']) #drop avoids duplication of index and dedicated column
    df_result['timestep_weights'] = weights_array
    df_result.to_csv(
        path_or_buf=f"simple_weather-year_ldes-model/data_tables/n_cluster_weather_years/cluster_timestep_weights_{'_'.join(str(x) for x in dict_input['weather_years'])}_{'_'.join(str(x) for x in dict_input['weights'])}.csv", 
        )
    
    df_result = pd.read_csv(f"simple_weather-year_ldes-model/data_tables/n_cluster_weather_years/cluster_timestep_weights_{'_'.join(str(x) for x in dict_input['weather_years'])}_{'_'.join(str(x) for x in dict_input['weights'])}.csv")

    return df_result

#build a model and modify the timestep weights
def run_model_with_timestep_weights(dict_input):

    #call functions to generate timeseries data_tables
    df_timestep_weights = generate_timestep_weights(dict_input)

    model = calliope.Model(
        'simple_weather-year_ldes-model/model_cluster_with_timestep_weights.yaml',
        scenario='single_year_runs_plan',
        override_dict={
             'config.init.time_subset': [df_timestep_weights['timesteps'].min(), df_timestep_weights['timesteps'].max()], 
             'techs.h2_salt_cavern.number_year_cycles': len(dict_input['weather_years']),
             'data_tables.time_varying_parameters.data': f"data_tables/n_cluster_weather_years/cluster_timeseries_{'_'.join(str(x) for x in dict_input['weather_years'])}.csv",
             }
        )
    
    model.inputs.timestep_weights.data=df_timestep_weights['timestep_weights'].to_numpy()
    print(f'Building Model [{','.join(str(x) for x in dict_input['weather_years'])}] with weights: [{':'.join(str(x) for x in dict_input['weights'])}]')
    model.build()
    print('Solving Model...')
    model.solve()
    # model.to_netcdf(f"simple_weather-year_ldes-model/results/n_year_cluster_runs/model_cluster_{'_'.join(str(x) for x in dict_input['weather_years'])}_{'_'.join(str(x) for x in dict_input['weights'])}.netcdf")

    return model

def n_cluster_model_path_parser(dict_input):
    return f"simple_weather-year_ldes-model/results/n_year_cluster_runs/model_cluster_{'_'.join(str(x) for x in dict_input['weather_years'])}_{'_'.join(str(x) for x in dict_input['weights'])}.netcdf"

def review_storage(path):

    model = calliope.read_netcdf(path)

    df_storage_cap = (
                    (model.results.storage_cap.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Hydrogen Storage Capacity (MWh)")
                    .reset_index()
                )
    
    print(df_storage_cap)

    df_storage = (
                    (model.results.storage.fillna(0))
                    .sel(techs="h2_salt_cavern")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("SOC (MWh)")
                    .reset_index()
                )

    fig_storage = px.area(
        df_storage,
        x="timesteps",
        y="SOC (MWh)",
        height=800,
    )

    fig_storage.write_html("simple_weather-year_ldes-model/results/n_year_cluster_runs/soc_graph.html", auto_open=True)

def review_highlights(path):

    output = {}
    model = calliope.read_netcdf(path)
    full_model = calliope.read_netcdf('simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf')
    
    df_storage_cap = (
                    (model.results.storage_cap.fillna(0))
                    .sel(techs="h2_salt_cavern")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Hydrogen Storage Capacity (MWh)")
                    .reset_index()
    )['Hydrogen Storage Capacity (MWh)'].max()

    df_storage_cap_full = (
                    (full_model.results.storage_cap.fillna(0))
                    .sel(techs="h2_salt_cavern")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Hydrogen Storage Capacity (MWh)")
                    .reset_index()
    )['Hydrogen Storage Capacity (MWh)'].max()
    
    df_cost = (
                    (model.results.cost.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Annualised Cost €M")
                    .reset_index()
    )["Annualised Cost €M"].sum()

    df_cost_full = (
                    (full_model.results.cost.fillna(0))
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Annualised Cost €M")
                    .reset_index()
    )["Annualised Cost €M"].sum()

    output = {
        'Parameter': [
            'H2 Storage (GWh)',
            "Annualised Cost €M",
            ],
        'Model Value': [
            df_storage_cap,
            df_cost,
            ],
        'Reference Value': [
            df_storage_cap_full,
            df_cost_full,
            ],
        'Error': [
            f"{round(100*(df_storage_cap/df_storage_cap_full-1),1)}%",
            f"{round(100*(df_cost/df_cost_full-1),1)}%",
            ]
        }
    
    techs_of_interest = ['solar','offshore_wind','electrolyser','h2_elec_conversion','battery']
    df_flow_cap_vals = []
    df_flow_cap_full_vals = []
    for tech in techs_of_interest:
        df_flow_cap = (
            (model.results.flow_cap.fillna(0))
            .sel(techs=tech)
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow Capacity (MW)")
            .reset_index()
        )
        df_flow_cap_full = (
            (full_model.results.flow_cap.fillna(0))
            .sel(techs=tech)
            .to_series()
            .where(lambda x: x != 0)
            .dropna()
            .to_frame("Flow Capacity (MW)")
            .reset_index()
        )
        output['Parameter'].append(f"{tech} Flow Cap (MW)")
        output['Model Value'].append(int(df_flow_cap["Flow Capacity (MW)"][0]))
        output['Reference Value'].append(int(df_flow_cap_full["Flow Capacity (MW)"][0])) 
        output['Error'].append(f"{round(((df_flow_cap["Flow Capacity (MW)"][0]/(df_flow_cap_full["Flow Capacity (MW)"][0])-1)*100).mean(),1)}%")

    return pd.DataFrame.from_dict(output).set_index('Parameter')

min_yr = 2010
max_yr = 2019

for i in range(min_yr,max_yr+1):
    for j in range (min_yr, max_yr+1):
        run = False

        if i != j and i+1 != j:
            if i != max_yr:

                input_dictionary ={
                    'weather_years': [j, i, i+1],
                    'weights': [8,1,1]
                }

                run = True

            else:

                input_dictionary ={
                    'weather_years': [j, i, min_yr],
                    'weights': [8,1,1]
                }

                if j != min_yr:
                    run = True
        
        if run:

            old_directory = 'simple_weather-year_ldes-model/results/n_year_cluster_runs'
            new_directory = 'simple_weather-year_ldes-model/results/three_year_cluster_runs'

            file = n_cluster_model_path_parser(input_dictionary)

            if os.path.isfile(n_cluster_model_path_parser(input_dictionary)):
                print('Result exists. Saving to new directory...', input_dictionary['weather_years'])
                # model = calliope.read_netcdf(file)
            else:
                print(input_dictionary['weather_years'])
                # model = run_model_with_timestep_weights(input_dictionary)
                

            # model.to_netcdf(f"simple_weather-year_ldes-model/results/three_year_cluster_runs/model_cluster_{'_'.join(str(x) for x in input_dictionary['weather_years'])}_{'_'.join(str(x) for x in input_dictionary['weights'])}.netcdf")


# run_model_with_timestep_weights(input_dictionary)