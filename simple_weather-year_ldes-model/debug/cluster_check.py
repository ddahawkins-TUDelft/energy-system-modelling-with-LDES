import tsam.timeseriesaggregation as tsam

import calliope



# Load data at full time resolution

model = calliope.Model(
    'weather_year_clustering_LDES_study/model_config/model.yaml',
)

# Get all timeseries data from model, with timesteps on the rows and all other dimensions on the columns

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

aggregation = tsam.TimeSeriesAggregation(

  raw_data, noTypicalPeriods=12, hoursPerPeriod=24, clusterMethod="hierarchical"

)

typPeriods = aggregation.createTypicalPeriods()

matched_indices = aggregation.indexMatching()

representative_dates = (

    raw_data

    .resample("1D")

    .first()

    .iloc[aggregation.clusterCenterIndices]

    .index

)

cluster_days = (

    matched_indices

    .resample("1D")

    .first()

    .PeriodNum

    .apply(lambda x: representative_dates[x])

)

cluster_days.to_csv("simple_weather-year_ldes-model/debug/clusters.csv")



model_clustered = calliope.Model(..., time_cluster="/absolute_path/to/clusters.csv")