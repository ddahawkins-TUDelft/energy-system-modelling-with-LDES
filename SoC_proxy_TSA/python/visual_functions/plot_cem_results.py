
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import calliope

from ..utility_functions.helper_calliope import read_clustered_netcdf
from ..utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
from ..utility_functions.helper_timeseries_tools import calliope_ts_to_pandas, extrapolate_ts_from_cluster_map

def cem_results(dictionary_model_ids, path_reference):

    # reference
    model_reference = calliope.read_netcdf(path_reference)

    x, y = get_capacities(model_reference)


    # for model_id, name in dictionary_model_ids:



def get_capacities(m: calliope.Model):

    df_power_caps = (m.results['flow_caps'].fillna(0).to_series().dropna()
          .to_frame('capacity').reset_index())
    
    df_energy_caps = (m.results['storage_caps'].fillna(0).to_series().dropna()
          .to_frame('capacity').reset_index())
    
    return 1,2
    

cem_results({},'SoC_proxy_TSA/data/calliope_models/standard_2015_2019_reference.nc')