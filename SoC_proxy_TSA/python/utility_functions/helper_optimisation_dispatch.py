from utility_functions.class_tsa_model import tsa_model
import pandas as pd



def optimisation_dispatch(m: tsa_model):

    result = {
        'cluster_map': pd.DataFrame
    }
    
    #check and validate the mode
    mode = m.tsa.params['soc_proxy']['optimisation_proxy_mode']
    if mode not in ['endogenous','exogenous']:
        raise Exception(f'{mode} is not a valid value for tsa.params[soc_proxy][optimisation_proxy_mode]. Options are [endogeneous,exogenous]')


    print('optimised')

    return result