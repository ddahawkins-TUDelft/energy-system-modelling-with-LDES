import tsam.timeseriesaggregation as tsam
import calliope

#function that receives calliope model and parameters, exports a tsam compressed timeseries
def apply_tsam_to_calliope_timeseries(model: calliope.Model, number_typical_periods: int, hours_per_period: int, cluster_method: str, path_to_cluster_csv: str):
    
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

    #perform tsam aggregation
    aggregation = tsam.TimeSeriesAggregation(
        raw_data, 
        noTypicalPeriods=number_typical_periods, 
        hoursPerPeriod=hours_per_period, 
        clusterMethod=cluster_method
    )

    #create typical periods from aggregation. Class, self-permutates so must be run.
    typPeriods = aggregation.createTypicalPeriods()

    # matching the indices of the aggregation output
    matched_indices = aggregation.indexMatching()

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

    cluster_days.to_csv(path_to_cluster_csv)

    #generate logging message
    log_message = f"TSA successfully applied. Results of {number_typical_periods} day {cluster_method} method saved to {path_to_cluster_csv}."

    return cluster_days, log_message