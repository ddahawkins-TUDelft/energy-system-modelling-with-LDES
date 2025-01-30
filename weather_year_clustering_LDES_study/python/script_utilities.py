import pandas as pd # PANDAS YAY
import ast  # To safely evaluate the string as a list
import numpy as np
import datetime


# Define a converter function for string to array
def string_to_list(value):
    try:
        return ast.literal_eval(value)  # Safely evaluate the string as a Python literal
    except (ValueError, SyntaxError):
        return value  # In case it's not a valid string for evaluation

#function for extracting case studies
def read_years_weights(model_type: str):

    # initilaise file path and set to default
    config_file_path = "weather_year_clustering_LDES_study/case_study_config/single_year_cluster.csv"

    #switch across model types and raise an exception if not valid. Returns a list of years and associated weights for model runs.
    if model_type=='single_year_cluster':
        return pd.read_csv("weather_year_clustering_LDES_study/case_study_config/single_year_cluster.csv", converters={'years': string_to_list,'weights': string_to_list, })
    if model_type=='two_year_cluster':
        return pd.read_csv("weather_year_clustering_LDES_study/case_study_config/two_year_clusters.csv", converters={'years': string_to_list,'weights': string_to_list, })
    if model_type=='three_year_cluster':
        return pd.read_csv("weather_year_clustering_LDES_study/case_study_config/three_year_clusters.csv", converters={'years': string_to_list,'weights': string_to_list, })
    else:
        raise Exception("Not a valid model type. Options are: single_year_cluster, two_year_cluster, three_year_cluster.")

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

#date validity checker
def is_valid_date(year, month, day):
    day_count_for_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if year%4==0 and (year%100 != 0 or year%400==0):
        day_count_for_month[2] = 29
    return (1 <= month <= 12 and 1 <= day <= day_count_for_month[month])


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
        path_or_buf=f"{dict_input['timeseries_save_path']}", 
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
    path_or_buf=dict_input['timeseries_weights_save_path'], 
    )
    
    df_result = pd.read_csv(dict_input['timeseries_weights_save_path'])


    return df_result

def generate_cluster(dict_input):

    dt = datetime.date(2010,1,1)
    dt_end = datetime.date(2019,12,31)
    step = datetime.timedelta(days=1)

    #find id of operational year
    id_op = dict_input['weights'].index(max(dict_input['weights']))
    yr_op = dict_input['weather_years'][id_op] #find operational year
    yrs_sd = [x for x in dict_input['weather_years'] if x != yr_op]

    all_dates = []
    representative_dates = []

    while dt < dt_end:

        date = dt.strftime('%Y-%m-%d')

        all_dates.append(date)

        #if statement to assign representative days
        if dt.year in yrs_sd:

            representative_dates.append(date) #if a system defining year, leave it alone

        else: #if any other year, set it to the operational year

            #check for leap day validity, if not, use 28th February instead
            if dt.day == 29 & dt.month == 2:
                if is_valid_date(yr_op, dt.month, dt.day):
                    representative_dates.append(datetime.date(yr_op,2,29).strftime('%Y-%m-%d'))  #if leap day exists in representative year, use it
                else: 
                    representative_dates.append(datetime.date(yr_op,2,28).strftime('%Y-%m-%d'))  #otherwise use 28 February
            else:
                representative_dates.append(datetime.date(yr_op,dt.month,dt.day).strftime('%Y-%m-%d')) #all other days of the year, use the representative day

        dt += step

    df = pd.DataFrame(
        {
        'timesteps': all_dates,
        'PeriodNum': representative_dates
        })

    return df