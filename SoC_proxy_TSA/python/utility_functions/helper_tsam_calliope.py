import tsam.timeseriesaggregation as tsam
import calliope
import pandas as pd

#function that receives calliope model and parameters, exports a tsam compressed timeseries
def apply_tsam_to_calliope_timeseries(model: calliope.Model, number_typical_periods: int, hours_per_period: int, cluster_method: str, rep_method: str, path_to_cluster_csv: str, path_to_new_timeseries: str):
    
    #extract timeseries from calliope model
    raw_data = (
    model.inputs[[
        k for k, v in model.inputs.data_vars.items()
        if "timesteps" in v.dims and len(v.dims) > 1
    ]]
    .to_dataframe()
    .stack()
    .unstack("timesteps")
    .T
    )
    save_columns = raw_data.columns
    old_names  = save_columns.names
    save_columns = pd.MultiIndex.from_tuples(
    [('comment',) + col for col in save_columns]
    )   
    save_columns.names = ['comment','nodes','techs','parameters']
    save_index = raw_data.index

    raw_data.columns = [col[1] if isinstance(col, tuple) else col for col in raw_data.columns]

    #perform tsam aggregation
    aggregation = tsam.TimeSeriesAggregation(
        raw_data, 
        noTypicalPeriods=number_typical_periods, 
        hoursPerPeriod=hours_per_period, 
        clusterMethod=cluster_method,
        representationMethod=rep_method
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
    how='left'  # or 'inner' depending on your need
    )

    df_new_timeseries_values.drop(['PeriodNum','TimeStep'], axis=1, inplace=True)

    df_new_timeseries_values.columns = save_columns
    df_new_timeseries_values.index = save_index.strftime('%Y/%m/%d %H:%M')

    if aggregation.clusterCenterIndices:
        #when using methods such as medoidRepresentation, we get an cluster center index which we can use to traceback the mapping of full res to clustered days

        #extract representative dates for embedding with new Calliope (clustered) model
        representative_dates = (
            raw_data
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
        periodnum_to_date = df.groupby('PeriodNum').apply(lambda x: x.index[0].date())

        # Step 4: Map PeriodNum to its first appearance date
        df['Date_map'] = df['PeriodNum'].map(periodnum_to_date)

        # Step 5: Reduce to just one row per date
        cluster_days = df[['date', 'Date_map']].drop_duplicates(subset='date')
        cluster_days = cluster_days.rename(columns={'date':'timesteps','Date_map': 'PeriodNum'}) #rename into a calliope friendly format
        
        # Step 6: Set index to date if desired
        cluster_days = cluster_days.set_index('timesteps')

    cluster_days.to_csv(path_to_cluster_csv)
    df_new_timeseries_values.to_csv(path_to_new_timeseries)

    #generate logging message
    log_message = f"TSA successfully applied. Results of {number_typical_periods} day {cluster_method} {rep_method} method saved to {path_to_cluster_csv}."

    return cluster_days, df_new_timeseries_values, log_message

