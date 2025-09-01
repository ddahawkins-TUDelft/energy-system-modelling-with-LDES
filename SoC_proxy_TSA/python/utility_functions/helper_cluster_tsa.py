import pandas as pd
import tsam.timeseriesaggregation as tsam
from typing import Dict, List, Tuple


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
    weightDict: dict | None = None,
    soc_features: dict | None = None
    ):

    #get the calliope export format for later export
    calliope_field_headings = pd.read_csv(path_to_new_timeseries, header=None, nrows=5)

    representationDict = None
    df_index = df_timeseries.index

    # add the daily magnitude features

    weightDict = dict(weightDict or {})
    added_cols_all: List[str] = []

    #scale soc_feature weights with proxy weights
    for key in soc_features.keys():
        soc_features[key] = soc_features[key]*weightDict[soc_proxy_dict["proxy_inputs_to_consider"][0]]


    if soc_features and soc_proxy_dict.get("use_soc_proxy"):
        proxy_cols = list(soc_proxy_dict.get("proxy_inputs_to_consider", []))
        multi_proxy = len(proxy_cols) > 1

        for proxy_col in proxy_cols:
            suffix = proxy_col if multi_proxy else None
            df_timeseries, new_weights, added_cols = _add_requested_soc_features(
                df_timeseries,
                delta_col=proxy_col,
                requested=soc_features,
                suffix=suffix,
            )
            weightDict.update(new_weights)
            added_cols_all.extend(added_cols)

        
        

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
    if soc_proxy_dict.get("use_soc_proxy"):
        # drop original proxy inputs
        df_new_timeseries_values.drop(
            columns=soc_proxy_dict.get("proxy_inputs_to_consider", []),
            errors="ignore",
            inplace=True,
        )
        # drop only the features we actually added
        if added_cols_all:
            df_new_timeseries_values.drop(columns=added_cols_all, errors="ignore", inplace=True)


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

def _add_requested_soc_features(
    df: pd.DataFrame,
    delta_col: str,
    requested: Dict[str, float],
    suffix: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Compute ONLY the requested SoC features from the hourly ΔSoC column `delta_col`,
    broadcast them to hourly resolution, and return:
        - augmented dataframe,
        - a weight dict mapping *actual column names* -> weight,
        - a list of added column names (for cleanup on export).
    If `suffix` is provided, feature columns are named '{feature}__{suffix}'.
    """
    d = df.copy()
    s = d[delta_col]
    by_day = s.resample("1D")

    # Base daily building blocks
    daily_net = by_day.sum()                                     # MWh/day (+ charge, - discharge)
    daily_dis = daily_net.clip(upper=0).abs()                    # discharge-only MWh/day

    # Define how to build each feature (daily series, then broadcast)
    def _name(f): return f"{f}__{suffix}" if suffix else f

    feature_builders = {
        "soc_activity_MWh":       lambda: by_day.apply(lambda x: x.abs().sum()).rename(_name("soc_activity_MWh")),
        "soc_charge_MWh":         lambda: by_day.apply(lambda x: x.clip(lower=0).sum()).rename(_name("soc_charge_MWh")),
        "soc_discharge_MWh":      lambda: by_day.apply(lambda x: (-x.clip(upper=0)).sum()).rename(_name("soc_discharge_MWh")),
        "soc_net_MWh":            lambda: daily_net.rename(_name("soc_net_MWh")),
        "soc_max_charge_rate":    lambda: by_day.max().rename(_name("soc_max_charge_rate")),
        "soc_max_discharge_rate": lambda: (-by_day.min()).rename(_name("soc_max_discharge_rate")),
        "soc_peak10h_charge":     lambda: by_day.apply(lambda x: x.rolling(10, min_periods=1).sum().max()).rename(_name("soc_peak10h_charge")),
        "soc_peak10h_discharge":  lambda: by_day.apply(lambda x: (-x).rolling(10, min_periods=1).sum().max()).rename(_name("soc_peak10h_discharge")),
        "soc_discharge_30d":      lambda: daily_dis.rolling(30,  min_periods=1).sum().rename(_name("soc_discharge_30d")),
        "soc_discharge_90d":      lambda: daily_dis.rolling(90,  min_periods=1).sum().rename(_name("soc_discharge_90d")),
        "soc_energy_debt":        lambda: _energy_debt(daily_net).rename(_name("soc_energy_debt")),
    }

    # Build only the requested features
    daily_feats = []
    actual_weights: Dict[str, float] = {}
    added_cols: List[str] = []

    for feat, wt in (requested or {}).items():
        if feat not in feature_builders:
            # silently ignore unknown features; you could log/print if preferred
            continue
        ser = feature_builders[feat]()   # a daily-indexed Series
        daily_feats.append(ser)
        actual_weights[ser.name] = float(wt)
        added_cols.append(ser.name)

    if daily_feats:
        daily_df = pd.concat(daily_feats, axis=1)

        # Broadcast daily scalars to each hour in that day (no 'key_0' artifacts)
        d = (
            d.assign(__date=d.index.floor("D"))
             .join(daily_df, on="__date")
             .drop(columns="__date")
        )

    return d, actual_weights, added_cols


def _energy_debt(daily_net: pd.Series) -> pd.Series:
    """Cumulative discharge 'debt' that resets when net >= 0 (simple drawdown metric)."""
    acc = 0.0
    out = []
    for x in daily_net:
        # x > 0 reduces debt, x < 0 increases debt
        acc = max(acc + (-x), 0.0)
        out.append(acc)
    return pd.Series(out, index=daily_net.index)