import pandas as pd
import calliope 
import numpy as np
from scipy.spatial.distance import cdist
import pyomo.environ as pyo
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import utility_functions.helper_timeseries_tools as tt
import matplotlib
matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
import matplotlib.pyplot as plt
from utility_functions.helper_model_config import clustered_model_config
import os
from sklearn.preprocessing import StandardScaler
from utility_functions.helper_compare_models import compare_models

def extract_timeseries_from_calliope(model: calliope.Model):
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
    

    raw_data.columns = [col[1] if isinstance(col, tuple) else col for col in raw_data.columns]

    # #apply the soc proxy for input into TSAM clustering
    # raw_data, capacity_factors, nominal_capacities = generate_soc_proxy(
    #         df=raw_data,
    #         demand_field='demand_power',
    #         renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
    #         dispatchable_techs=proxy_parameters['dispatchable_techs'],
    #         storage_process_losses=proxy_parameters['storage_process_losses'],
    #         soc_decomposition = proxy_parameters['soc_decomposition'],
    # )

    # #now drop the capacity factor, general surplus, dynamic soc, and SDES fields, instead retaining only the LDES Surplus field which serves as the static input: SoC stresses
    # raw_data.drop(columns=['mean_capacity_factor','surplus','surplus_SDES', 'soc_proxy_LDES', 'soc_proxy_SDES'], inplace=True)
    # raw_data.rename(columns={'surplus_LDES': 'soc_stresses'}, inplace=True)

    return raw_data

def generate_daily_feature_matrix(df: pd.DataFrame):
    """
    Restructure hourly time series data into a daily feature matrix where each row
    corresponds to a single day, and columns represent 24-hour profiles for each variable.

    Parameters:
    ----------
    df : pd.DataFrame
        Hourly time series with datetime index and one column per variable
        (e.g., demand_power, solar, onshore_wind, offshore_wind)

    Returns:
    -------
    pd.DataFrame
        Daily feature matrix with shape (n_days, n_features),
        where each row contains concatenated 24-hour profiles for each variable.
        Columns are named like 'demand_power_h00', ..., 'solar_h23'.
    """
    
    # Sanity check: ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Get the list of variables (columns)
    variables = df.columns.tolist()

    # Create a container for daily data
    daily_profiles = []

    # Group by day
    for day, group in df.groupby(df.index.date):
        if len(group) != 24:
            # Skip incomplete days (edge cases at start/end of dataset)
            continue
        day_vector = []
        for var in variables:
            # Add the full 24-hour profile for this variable
            day_vector.extend(group[var].values.tolist())
        daily_profiles.append(day_vector)

    # Build column names
    col_names = []
    for var in variables:
        for h in range(24):
            col_names.append(f"{var}_h{h:02d}")

    # Build DataFrame
    daily_df = pd.DataFrame(daily_profiles, columns=col_names)

    # Optional: Add date index (not required for MILP TSA)
    valid_days = [day for day, group in df.groupby(df.index.date) if len(group) == 24]
    daily_df.index = pd.to_datetime(valid_days)

    return daily_df

def aggregate_to_daily_features(df: pd.DataFrame, agg_func: str = "mean") -> pd.DataFrame:
    """
    Aggregate an hourly timeseries DataFrame into daily scalar features for TSA.

    Parameters:
    ----------
    df : pd.DataFrame
        Hourly timeseries with datetime index and multiple variable columns.

    agg_func : str
        Aggregation function to use per variable ('mean', 'sum', 'std', etc.).

    Returns:
    -------
    pd.DataFrame
        One row per day, with one aggregated value per variable.
        Columns are automatically inferred from the input.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Select only numeric columns (e.g., drop metadata if present)
    df_numeric = df.select_dtypes(include=[np.number])

    # Group by day and apply aggregation
    daily_df = df_numeric.resample("D").agg(agg_func)

    return daily_df

def compute_distance_matrix(feature_df: pd.DataFrame, metric: str = "euclidean"):
    """
    Compute the pairwise distance matrix between all daily feature vectors.

    Parameters:
    ----------
    feature_df : pd.DataFrame
        Daily feature matrix where each row corresponds to one day
        and columns are 24-hour profiles for each variable.

    metric : str
        Distance metric to use (any supported by scipy.spatial.distance.cdist).
        E.g., 'euclidean', 'manhattan', 'cosine'

    Returns:
    -------
    np.ndarray
        A 2D array of shape (n_days, n_days) representing pairwise distances.
    """

    X = feature_df.values  # shape: (n_days, n_features)
    distance_matrix = cdist(X, X, metric=metric)
    return distance_matrix

def compute_proxy_distance_matrix(feature_df: pd.DataFrame, proxy_parameters: dict, metric: str = "euclidean"):

    feature_df,_,_ = generate_soc_proxy(
            df=feature_df,
            demand_field='demand_power',
            renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
            dispatchable_techs=proxy_parameters['dispatchable_techs'],
            storage_process_losses=proxy_parameters['storage_process_losses'],
            soc_decomposition = proxy_parameters['soc_decomposition'],
            timestamp_col=None
    )
    
    source_cols = [col for col in feature_df.columns if col in ['solar', 'onshore_wind', 'offshore_wind']]
    demand_col = 'demand_power'
    surplus_col = 'surplus_LDES'

    renewables = feature_df[source_cols].copy()
    demand = feature_df[[demand_col]].copy()
    surplus = feature_df[[surplus_col]].copy()

    # Scale each group
    scaler_renew = StandardScaler()
    scaler_surplus = StandardScaler()
    scaler_demand = StandardScaler()

    X_renew_scaled = scaler_renew.fit_transform(renewables)
    X_surplus_scaled = scaler_surplus.fit_transform(surplus)
    X_demand_scaled = scaler_demand.fit_transform(demand)

    # Weight them (optional)
    weight_renew = 0.3
    weight_surplus = 0.4
    weight_demand = 0.3

    X_combined = np.hstack([
        weight_renew * X_renew_scaled,
        weight_surplus * X_surplus_scaled,
        weight_demand * X_demand_scaled
    ])

    # Compute distance matrix
    distance_matrix = cdist(X_combined, X_combined, metric=metric)

    return distance_matrix 

def solve_standard_milp_tsa(distance_matrix: np.ndarray, k: int, solver: str = "gurobi"):
    """
    Solve a MILP time series aggregation problem using Pyomo and a given distance matrix.

    Parameters:
    ----------
    distance_matrix : np.ndarray
        A symmetric matrix of shape (n_days, n_days) representing pairwise distances.
    k : int
        The number of representative days to select.
    solver : str
        MILP solver to use (e.g., "gurobi", "cbc", "glpk").

    Returns:
    -------
    dict
        Dictionary with:
            'selected_days': list of indices selected as representative days
            'assignments': dict mapping each day to its assigned representative
            'model': the Pyomo model instance (for inspection)
    """
    n_days = distance_matrix.shape[0]
    I = range(n_days)
    J = range(n_days)

    # Create model
    model = pyo.ConcreteModel()

    # Sets
    model.I = pyo.Set(initialize=I)
    model.J = pyo.Set(initialize=J)

    # Parameters
    model.D = pyo.Param(model.I, model.J, initialize=lambda model, i, j: distance_matrix[i, j])

    # Decision variables
    print(f'creating {n_days} binary variables for y[i]')
    model.y = pyo.Var(model.I, domain=pyo.Binary)  # y[i] = 1 if day i is selected
    print(f'creating {n_days*n_days} binary variables for x[i,j]')
    model.x = pyo.Var(model.I, model.J, domain=pyo.Binary)  # x[i,j] = 1 if day j is assigned to rep i

    # Objective: Minimize total distance
    print('Assigning objective function and constraints')
    def obj_rule(model):
        return sum(model.x[i, j] * model.D[i, j] for i in model.I for j in model.J)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # Constraints

    # Each day j must be assigned to exactly one representative i
    def assign_once_rule(model, j):
        return sum(model.x[i, j] for i in model.I) == 1
    model.assign_once = pyo.Constraint(model.J, rule=assign_once_rule)

    # x[i,j] can only be 1 if y[i] is selected
    def assign_only_if_selected_rule(model, i, j):
        return model.x[i, j] <= model.y[i]
    model.link_x_y = pyo.Constraint(model.I, model.J, rule=assign_only_if_selected_rule)

    # Limit number of representatives
    def num_reps_rule(model):
        return sum(model.y[i] for i in model.I) == k
    model.num_reps = pyo.Constraint(rule=num_reps_rule)

    # Solve
    print('Building and solving the model...')
    solver_obj = pyo.SolverFactory(solver)

    results = solver_obj.solve(model,
        tee=True,
        options={
            'LogToConsole': 1,     # Force Gurobi to show log
            'TimeLimit': 3000,      
            # 'MIPGap': 0.01         
        })

    # Extract solution
    selected_days = [i for i in model.I if pyo.value(model.y[i]) > 0.5]
    assignments = {j: next(i for i in model.I if pyo.value(model.x[i, j]) > 0.5) for j in model.J}

    return {
        "selected_days": selected_days,
        "assignments": assignments,
        "model": model,
        "results": results
    }

def save_milp_result_to_calliope_csv(
    result: dict,
    df_daily: pd.DataFrame,
    output_path: str
):
    """
    Save the MILP TSA result in the Calliope-compatible format:
    timesteps,PeriodNum

    Parameters
    ----------
    result : dict
        Output of `solve_standard_milp_tsa`, including 'assignments' and 'selected_days'.
    original_df : pd.DataFrame
        DataFrame used to generate the feature matrix, must have a DatetimeIndex.
    df_daily : pd.DataFrame
        DataFrame used to capture daily information.
    output_path : str
        Path to the CSV file to write.
    """
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily must have a DatetimeIndex")

    # Convert index to list so we can use integer positions
    daily_dates = df_daily.index.to_list()

    # Map day index → representative day
    day_to_rep = {
        daily_dates[day_idx]: daily_dates[result['assignments'][day_idx]]
        for day_idx in result['assignments']
    }

    # Build final dataframe
    mapping_df = pd.DataFrame.from_dict(day_to_rep, orient="index", columns=["PeriodNum"])
    mapping_df.index.name = "timesteps"
    mapping_df = mapping_df.reset_index()

    # Format as strings for Calliope compatibility
    mapping_df["timesteps"] = mapping_df["timesteps"].dt.strftime("%Y-%m-%d")
    mapping_df["PeriodNum"] = mapping_df["PeriodNum"].dt.strftime("%Y-%m-%d")

    # Save to CSV
    mapping_df.to_csv(output_path, index=False)
    print(f"Saved daily representative mapping to {output_path}")

def run_calliope_model_on_cluster(ref_model, reference_model, path_cluster_map, proxy_parameters):

    _, _, ref_soc = tt.get_capacities(reference_model)

    params = {
        'output_directory_name': 'tsa_dev',
        'output_model_name': 'clustered_2015_2019_customMilpTSA',
        'config_yaml_name': 'model',
        'horizon_start':  '2015-01-01',
        'horizon_end':  '2019-12-31',
        'filename_time_varying_parameters': 'full_horizon/time_varying_parameters',
        'calliope_full_log': [False, False],
        # 'dict_additional_overrides': {},
        'path_to_cluster_csv': path_cluster_map,
        'path_to_new_timeseries': 'original'
    }

    #compute soc proxy of reference model
    ref_df_soc_proxy = tt.calliope_ts_to_pandas('SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',f"{ref_model[:4]}-01-1",f"{ref_model[-4:]}-12-31")
    ref_df_soc_proxy, _, _ = generate_soc_proxy(
            df=ref_df_soc_proxy,
            demand_field='demand_power',
            renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
            dispatchable_techs=proxy_parameters['dispatchable_techs'],
            storage_process_losses=proxy_parameters['storage_process_losses'],
            soc_decomposition = proxy_parameters['soc_decomposition'],
            timestamp_col='timesteps'
    )
    ref_df_soc_proxy = ref_df_soc_proxy.set_index('timesteps')
    save_path = 'SoC_proxy_TSA/results/tsa_dev/clustered_2015_2019_customMilpTSA.netcdf'

    if os.path.exists(save_path):
        print(f"Skipping solving clustered model: {save_path} already exists.")
        clustered_model = calliope.read_netcdf(save_path)
    else:

        clustered_model = clustered_model_config(params)

        #build, solve, save clustered model
        print(f">> Building: {path_cluster_map}")
        clustered_model.build()
        #solve
        print(f">>> Solving: {save_path}")
        clustered_model.solve()
        #print results
        print(f">>> Results: Obj. Function: {clustered_model.results.cost.sum().item():e}, Solve Time: {round(clustered_model.results.timestamp_solve_complete - clustered_model.results.timestamp_solve_start,1)}s")
        #auto-save, first checks if full directory tree exists (if not, creates it)
        clustered_model.to_netcdf(save_path)
        print(f">>> Saved: {save_path}")

    cluster_df_soc_proxy = pd.DataFrame() #initialising these to be save
    cluster_df_soc = pd.DataFrame() #initialising these to be save

    #--------------------------------------------

    comparison_result = compare_models(
        model_reference = reference_model, 
        model_test =clustered_model, 
        df_clustermap_test_model=pd.read_csv(path_cluster_map))

    #process the clustering map
    cluster_map = pd.read_csv(path_cluster_map)
    cluster_map = cluster_map.rename(columns={
    'timesteps': 'datesteps',
    'PeriodNum': 'mapped_datesteps'
    })
    cluster_map['datesteps'] = pd.to_datetime(cluster_map['datesteps'], format='%Y-%m-%d')
    cluster_map['mapped_datesteps'] = pd.to_datetime(cluster_map['mapped_datesteps'], format='%Y-%m-%d')

    #pull the intracluster soc
    df_intracluster_soc = (   
            (clustered_model.results['storage'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('intra_soc')
            .reset_index()
        )
    df_intracluster_soc=df_intracluster_soc[df_intracluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = pd.to_datetime(df_intracluster_soc['timesteps'], format='%Y-%m-%d')


    #pull the intercluster soc
    df_intercluster_soc = (   
        (clustered_model.results['storage_inter_cluster'].fillna(0))
        .to_series()
        # .where(lambda x: x != 0)
        .dropna()
        .to_frame('inter_soc')
        .reset_index()
    )
    df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
    df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
    
    #merge everything and filter
    cluster_df_soc = df_intercluster_soc.merge(cluster_map, on='datesteps', how='left')
    cluster_df_soc = cluster_df_soc.merge(df_intracluster_soc, on='mapped_datesteps', how='left')
    cluster_df_soc = cluster_df_soc[['datesteps','timesteps','inter_soc','intra_soc']]

    #create a proper measure of timestamps
    time_only = cluster_df_soc['timesteps'].dt.time
    cluster_df_soc['full_timestamp'] = cluster_df_soc['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))    
    cluster_df_soc = cluster_df_soc.set_index('full_timestamp')


    #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
    cluster_df_soc['soc'] = cluster_df_soc['inter_soc']+cluster_df_soc['intra_soc']

    cluster_df_soc_proxy,_ = tt.extrapolate_ts_from_cluster_map(
        path_cluster_map,
        'SoC_proxy_TSA/data_tables/full_horizon/time_varying_parameters.csv',
    )

    cluster_df_soc_proxy, _, _ = generate_soc_proxy(
        df=cluster_df_soc_proxy,
        demand_field='demand_power',
        renewables_fields_and_weights=proxy_parameters['capacity_weights'], 
        dispatchable_techs=proxy_parameters['dispatchable_techs'],
        storage_process_losses=proxy_parameters['storage_process_losses'],
        soc_decomposition = proxy_parameters['soc_decomposition'],
        timestamp_col='timesteps'
    )
    cluster_df_soc_proxy = cluster_df_soc_proxy.set_index('timesteps')

    plt.figure(figsize=(12, 6))

    plt.plot(ref_soc.index, ref_soc['soc'], label=f"SoC, Reference, soc_peak={np.max(ref_soc['soc']):.1e} on {ref_soc['soc'].idxmax().strftime('%Y-%m-%d')}",color='black', zorder=102)  
    plt.plot(ref_df_soc_proxy.index, ref_df_soc_proxy['soc_proxy_LDES'], label=f"Proxy, Reference, soc_peak={np.max(ref_df_soc_proxy['soc_proxy_LDES']):.1e} on {ref_df_soc_proxy['soc_proxy_LDES'].idxmax().strftime('%Y-%m-%d')}", color='grey', zorder=101)   
    plt.plot(cluster_df_soc.index, cluster_df_soc['soc'], label="SoC, Cluster", color='red', zorder=1)
    plt.plot(cluster_df_soc_proxy.index, cluster_df_soc_proxy['soc_proxy_LDES'], label="SoC Proxy, Cluster", color='orange', zorder=1)

    plt.xlabel('Time')
    plt.ylabel('State of Charge')
    plt.title('Storage State of Charge Over Time')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def run():


    #load model
    path_full_model = 'SoC_proxy_TSA/results/tsa_dev/standard_2015_2019_reference.netcdf'
    full_model = calliope.read_netcdf(path_full_model)

    proxy_parameters = {
    'capacity_weights': {
        'solar': 1,
        'onshore_wind': .5, # making the baseline assumption of an even distribution between solar and wind -based products i.e. the sum of onshore and offshore wind equals solar
        'offshore_wind': .5 
    },
    'storage_process_losses': {
        'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
        'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
    },
    'dispatchable_techs': {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
        # 'relative_dispatchable': 0 #TODO: add in capacity for relative dispatchable
    },
    'soc_decomposition': {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    }

    }

    cluster_map_path = 'SoC_proxy_TSA/cache/cluster_maps/2015_2019_n_36_customMilpTSA.csv'

    if os.path.exists(cluster_map_path):
        print(f"Skipping MILP: {cluster_map_path} already exists.")
    else:

        #extract timeseries
        df_raw = extract_timeseries_from_calliope(full_model)
        # print(df_raw.head())   # See a sample of the output

        #restructure as daily feature sets
        df_daily = aggregate_to_daily_features(df_raw, agg_func="mean") #alternative that agrgegates to daily and drops the hourly info
        # df_daily = generate_daily_feature_matrix(df_raw)
        # print(df_daily.shape)

        #compute euclidiean distances
        # distance_matrix = compute_distance_matrix(df_daily, metric="euclidean")
        distance_matrix = compute_proxy_distance_matrix(df_daily, proxy_parameters, metric='euclidean')
        # print(distance_matrix.shape)  
        # print(distance_matrix[:3, :3])  # See a sample of the output
        print("Running MILP TSA...")
        result = solve_standard_milp_tsa(distance_matrix, k=36)
        save_milp_result_to_calliope_csv(
            result=result,
            df_daily=df_daily,  # this has hourly timesteps
            output_path=cluster_map_path
        )

    run_calliope_model_on_cluster('2015_2019', full_model, cluster_map_path, proxy_parameters)
    
    
run()