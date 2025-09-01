import pandas as pd
import tsam.timeseriesaggregation as tsam


def cluster_tsa(
    df_timeseries, number_typical_periods: int, 
    hours_per_period: int, 
    cluster_method: str, 
    rep_method: str, 
    path_to_cluster_csv: str,
    path_to_original_timeseries: str,
    path_to_new_timeseries: str, 
    soc_proxy_dict: dict):

    #get the calliope export format for later export
    calliope_field_headings = pd.read_csv(path_to_new_timeseries, header=None, nrows=5)

    representationDict = None
    df_index = df_timeseries.index


    #perform tsam aggregation
    aggregation = tsam.TimeSeriesAggregation(
        df_timeseries, 
        noTypicalPeriods=number_typical_periods, 
        hoursPerPeriod=hours_per_period, 
        clusterMethod=cluster_method,
        representationMethod=rep_method,
        representationDict=representationDict
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