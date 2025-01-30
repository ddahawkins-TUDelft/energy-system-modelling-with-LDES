import pandas as pd
import calliope
import numpy as np
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import math
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
import random


# Plan
# Import Timeseries
# Predictive Weighting Object
# Function, take threshold and apply RLE

def parser_ts_filepath(arr_three_years):
    # Generate filepath for given array of years
    directory = 'simple_weather-year_ldes-model/data_tables/n_cluster_weather_years'
    return f"{directory}/cluster_timeseries_{"_".join(str(x) for x in arr_three_years)}.csv"

def import_timeseries(filepath):

    # Read in the appropriate csv as pandas dataframe
    df_timeseries = pd.read_csv(filepath)

    df_timeseries.columns = ['timesteps','demand','solar','offshore_wind','onshore_wind']
    df_timeseries = df_timeseries[4:]
    # df_data.reset_index().set_index('timesteps')

    #change data types
    df_timeseries['timesteps'] = pd.to_datetime(df_timeseries['timesteps'], format='%Y/%m/%d %H:%M')
    df_timeseries['demand'] = pd.to_numeric(df_timeseries['demand'])
    df_timeseries['solar'] = pd.to_numeric(df_timeseries['solar'])
    df_timeseries['offshore_wind'] = pd.to_numeric(df_timeseries['offshore_wind'])
    df_timeseries['onshore_wind'] = pd.to_numeric(df_timeseries['onshore_wind'])

    return df_timeseries
    #rename columns and reformat data types

def parser_capacities_filepath(arr_three_years,arr_weights=[8,1,1]):
    # generate filepath given array of years and weights (default implied)
    directory = 'simple_weather-year_ldes-model/results/n_year_cluster_runs'
    return f"{directory}/model_cluster_{"_".join(str(x) for x in arr_three_years)}_{"_".join(str(x) for x in arr_weights)}.netcdf"

def extract_capacities(filepath):

    # Read in the model
    model = calliope.read_netcdf(filepath)
    
    # df of flow capacity results
    df = (
                    (model.results.flow_cap.fillna(0))
                    .sel(carriers="power")
                    .to_series()
                    .where(lambda x: x != 0)
                    .dropna()
                    .to_frame("Flow Capacity (Power) (MW)")
                    .reset_index()
                )
    return  df.loc[df["techs"].isin(['offshore_wind','onshore_wind','solar'])]

def RES_weights(df_capacities = pd.DataFrame()):

    dictionary = {'solar': 1, 'offshore_wind': 1, 'onshore_wind': 1, 'scale':50000}
    if not(df_capacities.empty) :
        # Normalise the capacities to the peak value in the set. These values act as the weights for the technology in the timeseries
        df_capacities['Normalised_max_cap_tech'] = df_capacities["Flow Capacity (Power) (MW)"] / (df_capacities["Flow Capacity (Power) (MW)"]).max()
        # if results array exists, import result capacities to define technology weights
        dictionary['solar'] = df_capacities[df_capacities['techs']=='solar']['Normalised_max_cap_tech'].mean()
        dictionary['offshore_wind'] = df_capacities[df_capacities['techs']=='offshore_wind']['Normalised_max_cap_tech'].mean()
        dictionary['onshore_wind'] = df_capacities[df_capacities['techs']=='onshore_wind']['Normalised_max_cap_tech'].mean()
        dictionary['scale'] = (df_capacities["Flow Capacity (Power) (MW)"]).max() # scale here corresponds with max tech capacity, to help identify the scale for setting a threshold
    
    return dictionary

def compute_run_length_encoding_coefficients(filepath_ts, filepath_capacities, threshold):
    df_timeseries = import_timeseries(filepath_ts) # import timeseries
    dict_technology_weights = RES_weights() #import model results #extract_capacities(filepath_capacities)

    # filter to ensure ts > 2010
    df_timeseries['timesteps'] = pd.to_datetime(df_timeseries['timesteps'])
    df_timeseries = df_timeseries[df_timeseries['timesteps'].dt.year >= 2010]

    # Normalise demand within timeseries
    df_timeseries['demand'] = df_timeseries['demand'] / (df_timeseries['demand']).mean()

    # Normalise RES capacity factors against div.demand and div.technology capacities
    df_timeseries['solar'] = df_timeseries['solar'] / df_timeseries['demand'] * dict_technology_weights['solar']
    df_timeseries['offshore_wind'] = df_timeseries['offshore_wind'] / df_timeseries['demand'] * dict_technology_weights['offshore_wind']
    df_timeseries['onshore_wind'] = df_timeseries['onshore_wind'] / df_timeseries['demand'] * dict_technology_weights['onshore_wind']

    #aggregate into single RES CF variable
    df_timeseries['mean_normalised_RES_cf'] = (df_timeseries['solar']+df_timeseries['offshore_wind']+df_timeseries['onshore_wind'])/3

    #apply threshold check
    df_timeseries['boolean_below_threshold'] = np.where(df_timeseries['mean_normalised_RES_cf'] < threshold, df_timeseries['mean_normalised_RES_cf'] - threshold, 0)
    df_timeseries['mnREScf_offset_threshold'] = df_timeseries['mean_normalised_RES_cf'] - threshold
    df_timeseries['accumulation_flow_differences'] = df_timeseries['mnREScf_offset_threshold'].cumsum()

    return df_timeseries.reset_index()

def solver_cf_threshold(filepath_ts,filepath_capacities, t_initial_guess):

    #config for residuals solver
    i = 0
    iter_lim = 200
    tolerance_limt = 1e-6 # Allowing a 1% error 
    time_solver_start = datetime.datetime.now()
    step_size_refactor_threshold = 0.1
    decimal_places = int(round(math.log10(abs(1/tolerance_limt)),0))

    # initial guess
    t_val = t_initial_guess #0.22675510591756204
    t_val_increment = 1e-8
    residual = compute_run_length_encoding_coefficients(filepath_ts, filepath_capacities,t_val)['accumulation_flow_differences'].iat[-1]
    t_val += t_val_increment


    print('Solving for RLE threshold that equates to a perfect year cycle...')
    # A crude solver to determine the threshold for over and under surplus accumulating across the year
    while i < iter_lim and abs(residual) > tolerance_limt :
        print('Duration: ', f"{round((datetime.datetime.now()-time_solver_start).total_seconds(),1)}s",'Iteration: ', i, 'residual: ', round(residual,decimal_places),'t_val: ',t_val)
        error = compute_run_length_encoding_coefficients(filepath_ts, filepath_capacities,t_val)['accumulation_flow_differences'].iat[-1]
        factor_difference = (error-residual)/residual
        if abs(factor_difference) <= step_size_refactor_threshold and abs(error) > 1/step_size_refactor_threshold :
            print('Increasing increment scale.')
            t_val_increment *= 1/step_size_refactor_threshold
        if abs(factor_difference) <= abs(t_val_increment):
            t_val_increment *= t_val_increment/factor_difference
        t_val_increment = t_val_increment*(1+factor_difference)
        t_val -= t_val_increment

        # reset iter conditions
        i += 1
        residual = error
    
    # reporting
    if i < iter_lim and abs(residual) < tolerance_limt:
        print('Solution found for t_val: ',t_val)
    if i < iter_lim and abs(residual) > tolerance_limt:
        print('Solution bounded by iter_max. Proceeding with t_val of: ',t_val)
    
    return t_val

def compute_timeseries(filepath_ts, filepath_capacities, t_initial_guess):

    t_val = solver_cf_threshold(filepath_ts,filepath_capacities, t_initial_guess)
    df = compute_run_length_encoding_coefficients(filepath_ts, filepath_capacities,t_val)

    return df, t_val

#generate plots
def generate_plots(dictionary_traces):

    df_trace_name = 'accumulation_flow_differences'
    
    fig = go.Figure()

    for key, parameter  in dictionary_traces.items():

        # add trace
        fig.add_trace(go.Scatter(
            x = parameter['model']['timesteps'],
            y = parameter['model'][df_trace_name],
            name = parameter['name'],
            marker_color = parameter['colour'][0]
        ))
    
        if parameter['show_stationary_points']:
            df_net_changes, df_maxima, df_minima = identifying_minima_maxima(parameter['model'], df_trace_name)
        
            fig.add_trace(go.Scatter(
                x = df_maxima['timesteps'],
                y = df_maxima[df_trace_name], 
                name = f'{parameter['name']}, Maxima',
                marker_color = parameter['colour'][1],
                mode='markers'
            ))

            fig.add_trace(go.Scatter(
                x = df_minima['timesteps'],
                y = df_minima[df_trace_name], 
                name = f'{parameter['name']}, Minima',
                marker_color = parameter['colour'][2],
                mode='markers'
            ))

            if parameter['show_station_point_changes']:

                fig.add_trace(go.Bar(
                    x = df_net_changes['timesteps'],
                    y = df_net_changes['net_value_change'],
                    offset=df_net_changes['net_time_offset'],
                    width=df_net_changes['net_timestep_change_ms'],
                    name = 'Net Difference Between Stationary Points',
                    marker_color = 'rgba(100,100,100,0.2)'

                ))

    # years = pd.to_datetime(df['timesteps']).dt.year.unique()
    # years = years[years>=2010]

    fig.update_layout(
        title=dict(text=f"RLE for Different Thresholds"),
        yaxis = dict(
            title = dict(
                text='Accumulation of deviations wrt threshold',
                font=dict(size=16)
            )
        ),
        # xaxis = dict(
        #     ticktext = years
        # ),
        legend=dict(
            # x=0,
            # y=1.0,
            bgcolor = 'rgba(255,255,255,1)',
            bordercolor='rgba(255, 255, 255, 1)',
            
        ),
        # plot_bgcolor= 'white',
        # yaxis_tickformat="5.0%",
    )
    fig.update_xaxes(nticks = 11, showgrid=True)

    
    fig.write_html('simple_weather-year_ldes-model/export/RLE_results.html', auto_open=True)
    # fig_net_changes.write_html('simple_weather-year_ldes-model/export/RLE_results_netchanges.html', auto_open=True)

def filter_lowpass(data: np.ndarray, cutoff: float, sample_rate: float = None, poles: int = 5):
    sos = signal.butter(poles, cutoff, 'lowpass', fs=sample_rate, output='sos')
    filtered_data = signal.sosfiltfilt(sos, data)
    return filtered_data

def filter_gaussian(data: np.ndarray, standard_deviation: int):

    # Apply Gaussian filter
    return gaussian_filter1d(data, standard_deviation) 

def signal_smoothing(s_to_filter, duration_lenth, order): #TODO:

    # savitsky golay filter
    s_output = signal.savgol_filter(s_to_filter, window_length=duration_lenth, polyorder=order)
    
    print('low pass')
    print('windowing > normalising > fir > maxima, from allaboutcircuits') #perhaps this method can be fine tuned by applying filters for cycles we know should exist, e.g. daily cycles, seasonal
   
    print('Gaussian filter')

    return s_output

def signal_decomposition():
    print('Empirical Mode Decomposition (EMD) or Hilbert-Huang Transform (HHT)')
    print('Fourier Transform')

def cyclical_analysis(signal):
    print('Peak to Peak Distance')
    print('Peak Amplitude')

def identifying_minima_maxima(df, df_trace_name):

    # switch off pd warning which is fair but not applicable here
    pd.options.mode.chained_assignment = None

    #ensure format of timesteps field
    df['timesteps'] = pd.to_datetime(df['timesteps'])

    #remove all minor peaks
    prominence_filter = 0.1*abs(max(df[df_trace_name].max(),df[df_trace_name].min()))

    index_maxima,_ = signal.find_peaks(df[df_trace_name], prominence=prominence_filter)
    df_maxima = df.loc[index_maxima]
    df_maxima['polarity'] = 'maxima'

    index_minima,_ = signal.find_peaks(-df[df_trace_name], prominence=prominence_filter)
    df_minima = df.loc[index_minima]
    df_minima['polarity'] = 'minima'

    #concatenate
    df_combined = pd.concat([df_maxima,df_minima]).sort_values(by=['timesteps'])

    #calculate net changes between stationary points

    df_net_movement = df_combined[['timesteps','accumulation_flow_differences','polarity']]

    #calculate first value difference in array
    diff_array = [df_net_movement['accumulation_flow_differences'].iloc[0]]
    #calculate first time difference in array
    time_diff_array = [(df_net_movement['timesteps'].iloc[0]-df['timesteps'].iloc[0]).total_seconds() *1e3]
    time_offset = [(df['timesteps'].iloc[0]-df_net_movement['timesteps'].iloc[0]).total_seconds()*1e3]

    for i in range(1,len(df_net_movement)):

        diff_array.append(df_net_movement['accumulation_flow_differences'].iloc[i]-df_net_movement['accumulation_flow_differences'].iloc[i-1])
        time_diff_array.append((df_net_movement['timesteps'].iloc[i]-df_net_movement['timesteps'].iloc[i-1]).total_seconds() *1e3)
        time_offset.append((df_net_movement['timesteps'].iloc[i-1]-df_net_movement['timesteps'].iloc[i]).total_seconds()*1e3)

    #setup a dictionary containing final change values between final stat point and end of timeseries
    final_change_dict = {
        'timesteps': df['timesteps'].iloc[-1],
        'net_value_change': 0-df_net_movement['accumulation_flow_differences'].iloc[-1],
        'net_timestep_change_ms': (df['timesteps'].iloc[-1]-df_net_movement['timesteps'].iloc[-1]).total_seconds()*1e3,
        'net_time_offset': (df_net_movement['timesteps'].iloc[-1]-df['timesteps'].iloc[-1]).total_seconds()*1e3,
    }

    df_net_movement['net_value_change'] = diff_array
    df_net_movement['net_timestep_change_ms'] = time_diff_array
    df_net_movement['net_time_offset'] = time_offset

    df_net_movement = pd.concat([df_net_movement,pd.DataFrame(final_change_dict, index=['0'])])

    # switch pd warning back on
    pd.options.mode.chained_assignment = 'warn'

    return df_net_movement, df_maxima, df_minima

def plot_general_chart():
    bool_three_cluster = False

    if bool_three_cluster:
        years = [2015,2016,2017]
        debug_filepath_ts = parser_ts_filepath(years)
        debug_filepath_capacities = parser_capacities_filepath(years)
        t_initial_guess = 0.1
    else:
        debug_filepath_ts = 'simple_weather-year_ldes-model/data_tables/time_varying_parameters.csv'
        debug_filepath_capacities = 'simple_weather-year_ldes-model/results/results_full_horizon_2010_2019.netcdf'
        t_initial_guess = 0.1

    df, t_val = compute_timeseries(debug_filepath_ts, debug_filepath_capacities,t_initial_guess)

    # df['accumulation_flow_differences'] =  signal_smoothing(df['accumulation_flow_differences'], 7680, 3)

    #normalise
    df['accumulation_flow_differences'] = df['accumulation_flow_differences'] / max(abs(df['accumulation_flow_differences'].min()),abs(df['accumulation_flow_differences'].max()))

    #lowpass filter
    df_lowpass = df.copy()
    df_lowpass['accumulation_flow_differences'] = filter_lowpass(df_lowpass['accumulation_flow_differences'],24,7680)

    #gaussian smoothing
    df_gaussian = df_lowpass.copy()
    df_gaussian['accumulation_flow_differences'] = filter_gaussian(df_gaussian['accumulation_flow_differences'],24*30)
    #renormalise post gaussian 
    # df_gaussian['accumulation_flow_differences'] = df_gaussian['accumulation_flow_differences'] / max(abs(df_gaussian['accumulation_flow_differences'].min()),abs(df_gaussian['accumulation_flow_differences'].max()))

    plot_dictionary = {
        'original' :
        {
            'name': 'Normalised',
            'model': df,
            'cf_threshold': t_val,
            'show_stationary_points': False,
            'show_station_point_changes' : False,
            'colour': ['rgb(51, 204, 204)']
        },
        'Lowpass Filtering' :
        {
            'name': 'Lowpass',
            'model': df_lowpass,
            'cf_threshold': t_val,
            'show_stationary_points': False,
            'show_station_point_changes' : False,
            'colour': ['rgb(102, 153, 255)']
        },
        'Gaussian Smoothing' :
        {
            'name': 'Gaussian Smoothing',
            'model': df_gaussian,
            'cf_threshold': t_val,
            'show_stationary_points': True,
            'show_station_point_changes' : True,
            'colour': ['black', 'red', 'orange']
        },
    }

    generate_plots(plot_dictionary)

def comparative_chart_plot():

    #establish seed years for review
    seed_years = [2010,2011,2012,2013,2014,2015,2016,2017] # TODO: 2018,2019

    #plot dictionary
    plot_dictionary = {}

    #create set defining array of years
    for i in seed_years:
        years = [i,(i+1 if i+1<=2019 else 2010+(i-2019)),(i+2 if i+2<=2019 else 2010+(i+1-2019))]

        #config
        filepath_ts = parser_ts_filepath(years)
        filepath_capacities = parser_capacities_filepath(years)
        t_initial_guess = 0.1

        # get result
        df, t_val = compute_timeseries(filepath_ts, filepath_capacities,t_initial_guess)

        #lowpass filter
        df_lowpass = df.copy()
        df_lowpass['accumulation_flow_differences'] = filter_lowpass(df_lowpass['accumulation_flow_differences'],24,7680)

        #gaussian smoothing
        df_gaussian = df_lowpass.copy()
        df_gaussian['accumulation_flow_differences'] = filter_gaussian(df_gaussian['accumulation_flow_differences'],24*30)

        name = f"[{",".join(str(x) for x in years)}]"

        #append to plot dictionary
        plot_dictionary[name] =  {
            'name': name,
            'model': df_gaussian,
            'cf_threshold': t_val,
            'show_stationary_points': True if name == '[2015,2016,2017]' else False,
            'show_station_point_changes' : True if name == '[2015,2016,2017]' else False,
            'colour': ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]),'red', 'orange']
        }

    generate_plots(plot_dictionary)
    

plot_general_chart()
# comparative_chart_plot()