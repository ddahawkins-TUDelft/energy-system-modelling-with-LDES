import pandas as pd
from datetime import datetime

def _format_standard_calliope_ts(df: pd.DataFrame):
    df['timesteps'] = pd.to_datetime(df['timesteps'])
    columns_to_make_numeric = df.columns.difference(['timesteps'])
    df[columns_to_make_numeric] = df[columns_to_make_numeric].apply(pd.to_numeric)

    return df

def calliope_ts_to_pandas(source: str, date_range_lower_bound: str="", date_range_upper_bound: str=""):
    # import timeseries
    df_timeseries = pd.read_csv(source)
    # reformat
    index_header = df_timeseries[df_timeseries.iloc[:,0].str.contains('techs')].index[0] #identify row containing new header
    df_timeseries.iloc[index_header,0]='timesteps'
    df_timeseries.columns=df_timeseries.iloc[index_header]
    df_timeseries = df_timeseries.iloc[(index_header+3):].reset_index(drop=True) #remove first row containing comment

    #ensure python reads values in the correct format
    df_timeseries = _format_standard_calliope_ts(df_timeseries)

    #filtering out all dates outside of desired range
    if date_range_lower_bound:
        df_timeseries= df_timeseries[df_timeseries['timesteps'] >= date_range_lower_bound]
    if date_range_upper_bound:
        df_timeseries= df_timeseries[df_timeseries['timesteps'] < (pd.to_datetime(date_range_upper_bound) + pd.Timedelta(days=1))]

    return df_timeseries
    
def extrapolate_ts_from_cluster_map(source_cluster_map: str, source_original_ts: str):

    #import clustering map and apply date format to all columns
    df = pd.read_csv(source_cluster_map)
    #capture proper date formats
    for col in df.columns:
        if df[col].dtype=='object':
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass #skips all columns that arent datetime, should be none.

    #import original timeseries
    df_original_ts = calliope_ts_to_pandas(source_original_ts)
    column_names_original = df_original_ts.columns.values

    #create date-only column in the original dataset  for joining to map
    df_original_ts['date'] = df_original_ts['timesteps'].dt.normalize()

    df = pd.merge(
        df,
        df_original_ts,
        how='left',
        left_on='PeriodNum',
        right_on='date'
    )

    #carry the time allocation from the aggregated date-time column to the original date column
    df['timesteps'] = pd.to_datetime(df['timesteps_x'].astype(str)+' '+df['timesteps_y'].dt.time.astype(str))

    #extract clustered datetimes in case that is of interest to the user
    s_clustered_datetimes = df['timesteps_y']

    #filter out the columns we are uninterested in exporting
    df = df[column_names_original]

    return df, s_clustered_datetimes

#convert datetime columns into hour of year
def convert_to_hour_of_year(timestamp_series):
    """
    Converts a Pandas Series of 'yyyy-mm-dd hh:mm:ss' strings to hour of the year (0–8759).
    
    Parameters:
        timestamp_series (pd.Series): Series of timestamps as strings.
        
    Returns:
        pd.Series: Series of integers representing the hour of the year.
    """
    # Convert to datetime
    dt = pd.to_datetime(timestamp_series)

    # Calculate hour of year
    hour_of_year = (dt - pd.Timestamp(year=dt.dt.year.min(), month=1, day=1)).dt.total_seconds() // 3600

    return hour_of_year.astype(int)

def hour_of_year_from_timestamp(timestamp):

    return int((timestamp - datetime(timestamp.year, 1, 1)).total_seconds() // 3600)