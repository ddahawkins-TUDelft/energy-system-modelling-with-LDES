import pandas as pd
import tsam.timeseriesaggregation as tsam


def cluster_tsa(
    df_timeseries: pd.DataFrame, 
    number_typical_periods: int, 
    hours_per_period: int, 
    cluster_method: str, 
    rep_method: str, 
    path_to_cluster_csv: str,
    path_to_original_timeseries: str,
    path_to_new_timeseries: str, 
    soc_proxy_dict: dict,
    weightDict: dict = None
    ):

    #get the calliope export format for later export
    calliope_field_headings = pd.read_csv(path_to_new_timeseries, header=None, nrows=5)

    representationDict = None
    df_index = df_timeseries.index

    use_daily_features = True

    # add the daily magnitude features
    if use_daily_features:
        for proxy_param in soc_proxy_dict['proxy_inputs_to_consider']:
            df_timeseries=add_daily_proxy_magnitude_features(df_timeseries, delta_col=proxy_param)

        daily_magntitude_features_weights = {
        "soc_activity_MWh": 0.0001,          # captures the busyness of the day via absolute throughput of both charge and discahrge
        "soc_charge_MWh": 1,            # captures total charged energy that day
        "soc_discharge_MWh": 0.0001,         # captures total discharge energy that day
        "soc_net_MWh": 1,               # captures net change
        "soc_max_charge_rate": 0.0001,     # captures peak charging power
        "soc_max_discharge_rate": 0.0001,  # captures peak discharging power
        "soc_peak10h_charge": 0.0001,       # emphasize sustained charging ramps over a 10h cycle
        "soc_peak10h_discharge": 0.0001     # emphasize sustained discharging ramps over a 10h cycle
        }
        weightDict.update(daily_magntitude_features_weights) 

    #perform tsam aggregation
    aggregation = tsam.TimeSeriesAggregation(
        df_timeseries, 
        noTypicalPeriods=number_typical_periods, 
        hoursPerPeriod=hours_per_period, 
        clusterMethod=cluster_method,
        representationMethod=rep_method,
        representationDict=representationDict,
        weightDict=weightDict
    )
    
    #create typical periods from aggregation. Class, self-permutates so must be run.
    typPeriods = aggregation.createTypicalPeriods()

    # matching the indices of the aggregation output
    matched_indices = aggregation.indexMatching()

    #goal is a df with timesteps as index 'yyyy-mm-dd', and PeriodNum as first/only field 'yyyy-mm-dd'
    # we have to export a new csv of time varying parameters because TSAM cannot provide a traceback to the original timeseries for all methods

    #produce the new csv by merging the above datasets
    typPeriods = typPeriods.reset_index()
    typPeriods = typPeriods.rename(columns={'level_0': 'PeriodNum'})

    df_new_timeseries_values = matched_indices.merge(
    typPeriods,
    on=['PeriodNum', 'TimeStep'],
    how='left' 
    )

    #drop the merging columns and also soc_stresses which did not exist in the original dataset
    df_new_timeseries_values.drop(['PeriodNum','TimeStep'], axis=1, inplace=True)
    if soc_proxy_dict['use_soc_proxy']:
        df_new_timeseries_values.drop(soc_proxy_dict['proxy_inputs_to_consider'], axis=1, inplace=True)

        if use_daily_features:
            for key in daily_magntitude_features_weights.keys():
                df_new_timeseries_values.drop(key, axis=1, inplace=True)


    df_new_timeseries_values.index = df_index.strftime('%Y/%m/%d %H:%M')

    if aggregation.clusterCenterIndices:
        #when using methods such as medoidRepresentation, we get an cluster center index which we can use to traceback the mapping of full res to clustered days

        #extract representative dates for embedding with new Calliope (clustered) model
        representative_dates = (
            df_timeseries
            .resample("1D")
            .first()
            .iloc[aggregation.clusterCenterIndices]
            .index
        )

        #apply the day clustering
        cluster_days = (
            matched_indices
            .resample("1D")
            .first()
            .PeriodNum
            .apply(lambda x: representative_dates[x])
        )
    else:
        #methods such as distributionRepresentation do not produce a clear mapping because they modify the time varying parameters, instead we have to create a new file and base our mapping on this

        # Step 1: Copy and ensure datetime index
        df = matched_indices.copy()
        df.index = pd.to_datetime(df.index)

        # Step 2: Add date column
        df['date'] = df.index.date

        # Step 3: Get mapping: PeriodNum → first date it appears
        df_reset = df.reset_index(names='timesteps')
        periodnum_to_date = df_reset.groupby('PeriodNum').first()['timesteps'].dt.date

        # Step 4: Map PeriodNum to its first appearance date
        df['Date_map'] = df['PeriodNum'].map(periodnum_to_date)

        # Step 5: Reduce to just one row per date
        cluster_days = df[['date', 'Date_map']].drop_duplicates(subset='date')
        cluster_days = cluster_days.rename(columns={'date':'timesteps','Date_map': 'PeriodNum'}) #rename into a calliope friendly format
        
        # Step 6: Set index to date if desired
        cluster_days = cluster_days.set_index('timesteps')
   
   #export CSVs, remembering to prepend the calliope bespoke fields for the timeseries export
    calliope_field_headings.to_csv(path_to_new_timeseries, index=False, header=False, mode="w")
    df_new_timeseries_values.to_csv(path_to_new_timeseries, index=True, header=False, mode="a")
    cluster_days.to_csv(path_to_cluster_csv)

    #generate logging message
    print(f">>> TSA successfully applied with SoC Proxy. Results saved to {path_to_cluster_csv}.")

    return cluster_days, df_new_timeseries_values

def add_daily_proxy_magnitude_features(df: pd.DataFrame, delta_col="surplus_LDES"):
    d = df.copy()
    by_day = d[delta_col].resample("1D")

    charge   = by_day.apply(lambda s: s.clip(lower=0).sum()).rename("soc_charge_MWh")
    discharge= by_day.apply(lambda s: (-s.clip(upper=0)).sum()).rename("soc_discharge_MWh")
    activity = by_day.apply(lambda s: s.abs().sum()).rename("soc_activity_MWh")
    net      = by_day.sum().rename("soc_net_MWh")
    max_ch   = by_day.max().rename("soc_max_charge_rate")
    max_dis  = (-by_day.min()).rename("soc_max_discharge_rate")

    # sustained 6h ramps (optional)
    def peak10h_pos(s): return s.rolling(10, min_periods=1).sum().max()
    def peak10h_neg(s): return (-s).rolling(10, min_periods=1).sum().max()
    peak6h_c = by_day.apply(peak10h_pos).rename("soc_peak10h_charge")
    peak6h_d = by_day.apply(peak10h_neg).rename("soc_peak10h_discharge")

    daily = pd.concat([charge, discharge, activity, net, max_ch, max_dis, peak6h_c, peak6h_d], axis=1)

    # broadcast to hours
    d = d.join(daily, on=d.index.floor("D"))
    d.drop(labels='key_0',axis=1, inplace=True)

    # optional robust scaling so weights are easier to tune
    # for c in daily.columns:
    #     q99 = d[c].quantile(0.99) or 1.0
    #     d[c] = d[c] / q99

    return d