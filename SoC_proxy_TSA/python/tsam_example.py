
import copy
import os
import pandas as pd
import matplotlib.pyplot as plt
import tsam.timeseriesaggregation as tsam

#import data
raw = pd.read_csv('SoC_proxy_TSA/cache/testdata.csv', index_col = 0)

#standard k-means aggregation
aggregation = tsam.TimeSeriesAggregation(
    raw, 
    noTypicalPeriods = 8, 
    hoursPerPeriod = 24,
    clusterMethod = 'hierarchical', 
    extremePeriodMethod = 'new_cluster_center',
    addPeakMin = ['T'], 
    addPeakMax = ['Load'] 
    )

print(aggregation.accuracyIndicators())

#create the typical periods
typPeriods = aggregation.createTypicalPeriods()

#export
typPeriods.to_csv('SoC_proxy_TSA/cache/outputdata.csv')