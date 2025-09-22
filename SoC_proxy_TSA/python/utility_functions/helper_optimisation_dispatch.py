from utility_functions.class_tsa_model import tsa
import pandas as pd
import numpy as np
import utility_functions.helper_optimisation_tsa as opt
from utility_functions.helper_cluster_tsa_with_extremes import ClusterResult


def optimisation_dispatch(
    tsa_config: tsa, 
    path_clustermap: str, 
    path_timeseries: str,
    feature_weights: dict,
    soc_proxy_params: dict,
    pre_cluster_result: ClusterResult = None,
    
    ):

    result = {
        'cluster_map': pd.DataFrame
    }

    df_features = tsa_config.df_features
    features = df_features.columns.values
    
    #check and validate the mode
    mode = tsa_config.params['soc_proxy']['optimisation_proxy_mode']
    if mode not in ['endogenous','exogenous']:
        raise Exception(f'{mode} is not a valid value for tsa.params[soc_proxy][optimisation_proxy_mode]. Options are [endogenous,exogenous]')

    use_soc_proxy = tsa_config.params['soc_proxy']['use_soc_proxy']
    proxy_inputs = tsa_config.params['soc_proxy']['proxy_inputs_to_consider']

    print('[TSA] Configuring MILP')

    # ======================== ENDOGENOUS MODE ======================== 
    if mode == 'endogenous':
        print(f'[TSA] Dispatching MILP with endogenous variables{" including "+" ".join(proxy_inputs) if proxy_inputs else ""}.')

        # ------------------------------ INPUT CHECKS -------------------------------------


        
        target_columns = tsa_config.params['name_demand']+tsa_config.params['names_renewables']

        if use_soc_proxy:

            #check all non-proxy inputs are present in dataframe
            missing = []
            for feature in target_columns:
                missing.append(feature) if feature not in features else None
            
            if missing:
                raise Exception(f'Mode is set to endogenous but the fields {missing} are missing from the dataframe of features')
            
            #setup the removal of exogenous soc proxy features
            extra = []
            for feature in proxy_inputs:
                extra.append(feature) if feature in features else None
            if extra:
                print(f'[TSA] In endogenous mode, all exogenous soc_proxy_features are removed from the optimisation as not to conflict with the endogenous generation of features. The following fields were removed from features dataframe prior to optimisation: {extra}')

            df_features = df_features[target_columns]
            
        else: 
            raise Exception('Mode cannot be set to endogenous whilst use_soc_proxy is set to false')
            
        
        
        # ------------------------------ Main -------------------------------------
                
        
        if pre_cluster_result: 
            
            # 1) Extract indices
            C, rep_for_day, dates_index, rep_dates_kept, missing = precluster_to_row_indices(
                df_features=tsa_config.df_features,                 # df features
                representatives=pre_cluster_result.representatives, # DatetimeIndex
                assignment=pre_cluster_result.assignment,           # Series (optional)
            )

            fix_reps_flag = (len(C) == tsa_config.params['k_periods'])


            # pull hourly surplus (N x 24) from cache, same day order as df_features (daily)
            S_by_day = tsa_config._surplus_hourly_by_day
            if S_by_day is None or S_by_day.shape[0] != len(tsa_config.df_features):
                raise RuntimeError("Endogenous-restricted path needs cached hourly surplus (N x 24) aligned with df_features.")
            

            eta_ch  = float(soc_proxy_params['storage_process_losses']['charging_efficiency'])
            eta_dis = float(soc_proxy_params['storage_process_losses']['discharging_efficiency'])
            lambda_soc = float(tsa_config.params['soc_proxy'].get('lambda_soc', 0.5))

            result = opt.solve_ordo_with_endogenous_soc_restricted(
                df_features=tsa_config.df_features,        # DAILY
                k=tsa_config.params['k_periods'],
                feature_weights=feature_weights,
                preferred_features=target_columns,         # same set as exogenous ORDO
                candidates=C,                              # GLOBAL ids from precluster
                fix_reps=fix_reps_flag,                    # True if |C|==k → pure reassignment
                warm_start_rep_for_day=rep_for_day,        # GLOBAL per-day rep, feasible start
                eta_ch=eta_ch,
                eta_dis=eta_dis,
                lambda_soc=lambda_soc,
                surplus_hourly_by_day=S_by_day,            # (N, 24)
                normalize="minmax_signed",
                solver="gurobi",
                MIPGap=tsa_config.params.get('mipgap', 0.01),
                threads=10,
                timelimit=tsa_config.params.get('timelimit', 3600),
                verbose=True,
                lp_method="barrier",
            )


        else:
            print('[TSA] Solving MILP with endogenous soc proxy features')



            eta_ch  = float(soc_proxy_params['storage_process_losses']['charging_efficiency'])
            eta_dis = float(soc_proxy_params['storage_process_losses']['discharging_efficiency'])
            lambda_soc = float(tsa_config.params['soc_proxy'].get('lambda_soc', 0.5)) 

            #pull the cached (N_days x 24) surplus directly ---
            S_by_day = tsa_config._surplus_hourly_by_day  # set in class_tsa_model per above
            if S_by_day is None or S_by_day.size == 0:
                raise RuntimeError("Endogenous mode requires cached hourly surplus; none found. "
                                "Ensure _build_timeseries cached it (optimisation + endogenous).")
            reference_surplus = S_by_day.reshape(-1)  # (T,)

            result = opt.solve_ordo_with_endogenous_soc(
                df_features=tsa_config.df_features,
                k=tsa_config.params['k_periods'],
                feature_weights=feature_weights,
                preferred_features=None,
                surplus_columns_24h=None,           
                eta_ch=eta_ch,
                eta_dis=eta_dis,
                lambda_soc=lambda_soc,
                reference_surplus=reference_surplus,
                normalize="minmax_signed",
                surplus_hourly_by_day=S_by_day,
            )

        print('[TSA] Solution Found')

        #get index for saving
        dates_index = df_features.resample("D").agg('mean').index

        opt.save_milp_result_to_cluster_map(
                result=result, 
                dates_index=dates_index,
                output_path=path_clustermap
            )
        
        print(f'[TSA] Saving cluster map to {path_clustermap}')
        
        return result
    
    # ======================== EXOGENOUS MODE ======================== 
    elif mode == 'exogenous':

        # ------------------------------ INPUT CHECKS -------------------------------------

        #columns for the exogenous solver to consider, if use_soc_proxy is true, these additional columns will be captured next
        target_columns = tsa_config.params['name_demand']+tsa_config.params['names_renewables']

        if use_soc_proxy:
            missing = []
            for feature in proxy_inputs:
                missing.append(feature) if feature not in features else None
            
            if missing:
                raise Exception(f'Mode is set to exogenous but the fields are {missing} are missing from the dataframe of features')

            target_columns.extend(proxy_inputs)
        else: 

            extra = []
            for feature in features:
                extra.append(feature) if feature not in target_columns else None
            if extra:
                print(f'[TSA] use_soc_proxy was set to False. The following fields were removed from features dataframe prior to optimisaiton: {extra}')

        
        # ------------------------------ Main ------------------------------------- 
        
        if pre_cluster_result: 

            # 1) Extract indices
            C, rep_for_day, dates_index, rep_dates_kept, missing = precluster_to_row_indices(
                df_features=tsa_config.df_features,                 # df features
                representatives=pre_cluster_result.representatives, # DatetimeIndex
                assignment=pre_cluster_result.assignment,           # Series (optional)
            )

            fix_reps_flag = (len(C) == tsa_config.params['k_periods'])

            result = opt.solve_ordo_from_features_restricted(
                df_features=tsa_config.df_features,
                k=tsa_config.params['k_periods'],
                feature_weights=feature_weights,
                preferred_features=target_columns,     # as you already use for ORDO
                normalize="minmax_signed",
                candidates=C,
                fix_reps=fix_reps_flag,
                warm_start_rep_for_day=rep_for_day,    # <- feasible MIP start from clustering
                lp_method="barrier",
            )
            
        else:

            print('[TSA] Solving MILP using exogenous features only')
            result = opt.solve_ordo_from_features(
                df_features=tsa_config.df_features,
                k=tsa_config.params['k_periods'],
                feature_weights=feature_weights,
                preferred_features=target_columns,        # or pass a subset like ['demand_power','onshore_wind', ...]
                normalize="minmax_signed",
            )
        print('[TSA] Solution Found')

        #get index for saving
        dates_index = df_features.resample("D").agg('mean').index

        opt.save_milp_result_to_cluster_map(
                result=result, 
                dates_index=dates_index,
                output_path=path_clustermap
            )
        
        print(f'[TSA] Saving cluster map to {path_clustermap}')
        
        return result


    else:
        raise Exception('Invalid mode.')

    return result


def _normalize_dates_like(x: pd.Index | pd.Series) -> pd.DatetimeIndex:
    """
    Ensure tz-naive daily datetimes for safe equality & lookup.
    """
    dx = pd.to_datetime(x, errors="coerce")
    # strip time & tz
    if isinstance(dx, pd.DatetimeIndex):
        dx = dx.tz_localize(None) if dx.tz is not None else dx
        return dx.normalize()
    else:
        dx = dx.dt.tz_localize(None) if getattr(dx.dt, "tz", None) is not None else dx
        return dx.dt.normalize()

def precluster_to_row_indices(
    df_features: pd.DataFrame,
    representatives: pd.DatetimeIndex,
    assignment: pd.Series | None = None,
):
    """
    Map pre-cluster outputs to row indices of df_features.

    Parameters
    ----------
    df_features : DataFrame with one row per day (index should be daily dates or convertible)
    representatives : DatetimeIndex of representative days
    assignment : Series mapping original day -> representative day (optional; for hard RR export)

    Returns
    -------
    C : list[int]                 # row indices for reps (order matches 'representatives')
    rep_for_day : np.ndarray|None # N-vector of rep row index per day (for hard RR)
    dates_index : pd.DatetimeIndex# normalized daily index for df_features (useful elsewhere)
    rep_dates_kept : pd.DatetimeIndex # reps actually found in df_features (same length as C)
    missing_reps : list[pd.Timestamp]  # reps that were not found (if any)
    """
    # normalize df_features index to daily
    if not isinstance(df_features.index, pd.DatetimeIndex):
        dates_index = _normalize_dates_like(df_features.index)
        df_features = df_features.copy()
        df_features.index = dates_index
    else:
        dates_index = _normalize_dates_like(df_features.index)

    # build lookup: date -> row integer
    row_of_date = {d: i for i, d in enumerate(dates_index)}

    # normalize representatives and map to row ids
    reps_norm = _normalize_dates_like(representatives)
    C = []
    rep_dates_kept = []
    missing_reps = []
    for d in reps_norm:
        i = row_of_date.get(d, None)
        if i is None:
            missing_reps.append(d)
        else:
            C.append(i)
            rep_dates_kept.append(d)
    rep_dates_kept = pd.DatetimeIndex(rep_dates_kept)

    rep_for_day = None
    if assignment is not None:
        # normalize assignment index & values
        asg_idx = _normalize_dates_like(assignment.index)
        asg_val = _normalize_dates_like(assignment)

        # initialize with -1 to detect holes
        N = len(df_features)
        rep_for_day = np.full(N, -1, dtype=int)

        # build a fast map for rep day -> its row index
        rep_row = {d: row_of_date[d] for d in rep_dates_kept}

        # iterate through assignment; map each original day (row j) to rep row i
        for d_day, d_rep in zip(asg_idx, asg_val):
            j = row_of_date.get(d_day, None)
            i = rep_row.get(d_rep, row_of_date.get(d_rep, None))
            if j is not None and i is not None:
                rep_for_day[j] = i

        # if any remain -1 (e.g., missing in df_features), fallback: nearest calendar date or first rep
        if (rep_for_day < 0).any() and len(C) > 0:
            fallback = C[0]
            rep_for_day[rep_for_day < 0] = fallback

    return C, rep_for_day, dates_index, rep_dates_kept, missing_reps
