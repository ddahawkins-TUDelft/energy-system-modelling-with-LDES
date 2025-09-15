from utility_functions.class_tsa_model import tsa
import pandas as pd
import utility_functions.helper_optimisation_tsa as opt


def optimisation_dispatch(tsa_config: tsa, path_clustermap: str):

    result = {
        'cluster_map': pd.DataFrame
    }

    df_features = tsa_config.df_features
    features = df_features.columns.values
    
    #check and validate the mode
    mode = tsa_config.params['soc_proxy']['optimisation_proxy_mode']
    if mode not in ['endogenous','exogenous']:
        raise Exception(f'{mode} is not a valid value for tsa.params[soc_proxy][optimisation_proxy_mode]. Options are [endogeneous,exogenous]')

    use_soc_proxy = tsa_config.params['soc_proxy']['use_soc_proxy']
    proxy_inputs = tsa_config.params['soc_proxy']['proxy_inputs_to_consider']


    



    #if endogenous without proxy, or exogenous, make sure to remove any additional fields
    if  (mode=='exogenous'):

        target_columns = tsa_config.params['name_demand']+tsa_config.params['names_renewables']

        extra = []
        for feature in features:
            extra.append(feature) if feature not in target_columns else None
        if extra:
            print(f'[TSA]: The fields {extra} were removed from features dataframe prior to optimisaiton.')

    # ======================== ENDOGENOUS MODE ======================== 
    if mode == 'endogenous':
        print(f'[TSA] Dispatching MILP with endogenous variables{" including "+" ".join(proxy_inputs) if proxy_inputs else ""}.')

        # ------------------------------ INPUT CHECKS -------------------------------------
        if use_soc_proxy:
            missing = []
            for feature in proxy_inputs:
                missing.append(feature) if feature not in features else None
            
            if missing:
                raise Exception(f'Mode is set to endogenous but the fields are {missing} are missing from the dataframe of features')
        
        else: 
            target_columns = tsa_config.params['name_demand']+tsa_config.params['names_renewables']

            extra = []
            for feature in features:
                extra.append(feature) if feature not in target_columns else None
            if extra:
                print(f'[TSA]: The fields {extra} were removed from features dataframe prior to optimisaiton.')
        
        # ------------------------------ Main -------------------------------------

        tsa_config.distance_matrix = opt.distance_matrix(
                feature_df= tsa_config.df_features,
                matrix_weights=tsa_config.params['matrix_weights'],
                metric=tsa_config.params['distance_matrix_metric'],
                column_prefixes_renewables=tsa_config.params['names_renewables'],
                column_prefixes_demand=tsa_config.params['name_demand'],
                column_prefixes_proxy=tsa_config.params['soc_proxy']['proxy_inputs_to_consider'] if tsa_config.params['soc_proxy']['use_soc_proxy'] else [],
                proxy_window = tsa_config.params['soc_proxy']['proxy_window'] if tsa_config.params['soc_proxy']['use_soc_proxy'] else None
            )
        

        print('[TSA] Solving MILP')
        result = opt.milp_tsa(
            distance_matrix=tsa_config.distance_matrix,
            k=tsa_config.params['k_periods'],
            solver='gurobi',
            mipgap=0.01,
            verbose=True
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
        if not use_soc_proxy :
            raise Exception('Mode cannot be set to exogenous whilst use_soc_proxy is set to false')
        
        target_columns = tsa_config.params['name_demand']+tsa_config.params['names_renewables']

        extra = []
        for feature in features:
            extra.append(feature) if feature not in target_columns else None
        if extra:
            print(f'[TSA]: The fields {extra} were removed from features dataframe prior to optimisaiton.')
        
        # ------------------------------ Main -------------------------------------

        



    else:
        raise Exception('Invalid mode.')

    return result