import pandas as pd
import numpy as np
from .helper_normalise import normalise_by_method
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from pandas import Timestamp
from scipy.signal import convolve
from numpy.fft import fft, ifft, fftfreq


#function that takes a proxy and demand data, attempts to resolve any scalar differences between a real world state of charge and proxy by determining storage system discharge during renewable scarcity events
def proxy_surplus_energy(df_timeseries: pd.DataFrame, demand_field: str, demand_scalar: float):
    surplus_factor = df_timeseries['surplus']
    surplus_energy = surplus_factor*demand_scalar 
    return surplus_energy 

def proxy_scalar_weighted_scarcity_surplus(df_timeseries: pd.DataFrame, demand_field: str, demand_scalar: float, alpha: float, quantile_threshold: float):
   

    # Scarcity tail
    q_low = quantile_threshold
    threshold_low = df_timeseries['surplus'].quantile(q_low)
    scarcity_df = df_timeseries[df_timeseries['surplus'] <= threshold_low]
    scarcity_scalar = np.average(
        np.abs(scarcity_df['surplus'] * scarcity_df[demand_field]),
        weights=scarcity_df[demand_field]
    )

    # Surplus tail 
    q_high = 1 - quantile_threshold
    threshold_high = df_timeseries['surplus'].quantile(q_high)
    surplus_df = df_timeseries[df_timeseries['surplus'] >= threshold_high]
    surplus_scalar = np.average(
        np.abs(surplus_df['surplus'] * surplus_df[demand_field]),
        weights=surplus_df[demand_field]
    )

    # Combine both
    combined_scalar = 1 / ((scarcity_scalar + surplus_scalar) / 2)


    surplus_energy = proxy_surplus_energy(df_timeseries,demand_field,demand_scalar)*(combined_scalar)

    return surplus_energy 

#function that generates the State of Charge proxy given a pandas dataframe containing timeseries information, meta data, and some system config data
# def generate_SoC_proxy(
#     df_timeseries: pd.DataFrame, 
#     demand_field: str='demand_power', 
#     renewables_fields_and_weights: dict = {
#         'solar':1,
#         'onshore_wind': 1,
#         'offshore_wind': 1
#         }
#     ):

#     #Quick process for checking all the fields exist and also gathering the names of the renewables
#     missing_fields = []
#     renewable_fields = []
#     if  demand_field not in df_timeseries.columns:
#         missing_fields.append(demand_field)
#     for key in renewables_fields_and_weights:
#         renewable_fields.append(key) 
#         if key not in df_timeseries:
#             missing_fields.append(key)
#     if missing_fields:
#         raise Exception(f"The following fields are missing from the dataframe: {missing_fields}")

#     #normalise demand by mean
#     df_timeseries[demand_field], demand_scaling_factor = normalise_by_method(df_timeseries[demand_field], method='mean')

#     #normalise capacity factors by sum, for unity across summation
#     renewables_fields_and_weights, _ = normalise_by_method(renewables_fields_and_weights, method='sum')

#     #For each timestep, divide the renewables capacity factors by the normalised demand for that timestep, to get capacity factors relative to demand
#     #This seeks to capture events where renewable output is low, but so is demand, i.e. no issues with supply.
#     df_timeseries[renewable_fields] = df_timeseries[renewable_fields].div(df_timeseries[demand_field], axis=0)

#     #compute the threshold between charge states and discharge states that govern the State of Charge proxy
#     threshold_SoC = 0
#     for i in renewable_fields:
#         tech_mean_cf = df_timeseries[i].mean()
#         tech_weight = renewables_fields_and_weights[i]

#         threshold_SoC += tech_mean_cf*tech_weight
    
#     #compute surplus estimates using renewable profiles and relative weights
#     df_timeseries['surplus'] = (
#         sum(df_timeseries[field] * renewables_fields_and_weights[field] for field in renewable_fields)
#         - threshold_SoC
#     )

#     #correct for floating point errors
#     mask = np.isclose(df_timeseries['surplus'], 0.0, atol=1e-12)
#     df_timeseries.loc[mask, 'surplus'] = 0.0

#     #compute the cumulation of surplus i.e. the state of charge proxy
#     df_timeseries['soc_proxy'] = df_timeseries['surplus'].cumsum()

#     #correct for floating point errors
#     mask = np.isclose(df_timeseries['soc_proxy'], 0.0, atol=1e-12)
#     df_timeseries.loc[mask, 'soc_proxy'] = 0.0
    

#     # df_timeseries['soc_proxy_scaled'] = (proxy_surplus_energy(df_timeseries, demand_field, demand_scaling_factor)).cumsum()
#     df_timeseries['surplus_scaled'] = proxy_scalar_weighted_scarcity_surplus(df_timeseries, demand_field, demand_scaling_factor, threshold_SoC, 0.01)

#     # debug_val_1 = df_timeseries['surplus_scaled'].iloc[0]
#     # debug_val_2 = df_timeseries['surplus_scaled'].iloc[-1]
#     # debug_val_3 = np.sum(df_timeseries['surplus_scaled'])

#     df_timeseries['soc_proxy_scaled'] = df_timeseries['surplus_scaled'].cumsum()

#     # debug_val_5 = df_timeseries['soc_proxy_scaled'].iloc[0]
#     # debug_val_6 = df_timeseries['soc_proxy_scaled'].iloc[-1]
    

#     #apply a correction for floating point noise i.e. we want a value of 0 at the end not #1e-9
#     numeric_columns = df_timeseries.select_dtypes(include=[np.number]).columns #gather all the numeric columns including the ones generated by this function to apply correction
#     arr = df_timeseries[numeric_columns].to_numpy()
#     arr[np.isclose(arr, 0, atol=1e-9)] = 0 #threshold set to 1e-9. Could be tighter but this seems adequate for a proxy.
#     df_timeseries[numeric_columns] = arr 

#     #corrections for comparison with real SoC, translate graph such that storage is never below 0
#     df_timeseries['soc_proxy'] = df_timeseries['soc_proxy']+np.abs(np.min(df_timeseries['soc_proxy']))
#     df_timeseries['soc_proxy_scaled'] = df_timeseries['soc_proxy_scaled']+np.abs(np.min(df_timeseries['soc_proxy_scaled']))

    

#     #return the timeseries with new SoC Proxy as well as the threshold
#     return df_timeseries, threshold_SoC, demand_scaling_factor

# def generate_soc_proxy_2(
#     df_timeseries: pd.DataFrame, 
#     demand_field: str='demand_power', 
#     renewables_fields_and_weights: dict = {
#         'solar':1,
#         'onshore_wind': 1,
#         'offshore_wind': 1
#         },
#     storage_process_losses: dict = {
#     'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
#     'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
#     }
#     ):

#     #Quick process for checking all the fields exist and also gathering the names of the renewables
#     missing_fields = []
#     renewable_fields = []
#     if  demand_field not in df_timeseries.columns:
#         missing_fields.append(demand_field)
#     for key in renewables_fields_and_weights:
#         renewable_fields.append(key) 
#         if key not in df_timeseries:
#             missing_fields.append(key)
#     if missing_fields:
#         raise Exception(f"The following fields are missing from the dataframe: {missing_fields}")

#     # #normalise demand by mean
#     # df_timeseries[demand_field], demand_scaling_factor = normalise_by_method(df_timeseries[demand_field], method='mean')

#     #normalise capacity factors by sum, for unity across summation
#     renewables_fields_and_weights, _ = normalise_by_method(renewables_fields_and_weights, method='sum')

#     #compute the weighted mean capacity factor for each technology and the combined value
#     weighted_mean_capacity_factors = {}
#     for tech in renewable_fields:
#         weighted_mean_capacity_factors[tech] = df_timeseries[tech].mean()*renewables_fields_and_weights[tech]
#     weighted_mean_capacity_factors['combined'] = sum(weighted_mean_capacity_factors.values())
#     #and the weighted series of capacity factors
#     df_timeseries['mean_capacity_factor'] = df_timeseries[list(renewables_fields_and_weights)].mul(pd.Series(renewables_fields_and_weights)).sum(axis=1)

#     #compute mean installed capacity
#     weighted_mean_installed_capacity = np.sum(df_timeseries[demand_field]) / np.sum(df_timeseries['mean_capacity_factor'])

#     #compute surpluses and cumulative state of charge
#     df_timeseries['surplus'] = df_timeseries['mean_capacity_factor']*weighted_mean_installed_capacity-df_timeseries[demand_field]
#     df_timeseries['soc_proxy'] = df_timeseries['surplus'].cumsum()
    
#     #correction for technology losses:
#     # !LIMITATION! Ideally, we would factor these efficiencies into the surplus calculations, but this breaks the linearity of the problem and it can no longer be solved analytically, requiring a numerical method
#     # This defeats the goal of the proxy which is to be lightweight. Here we use Round Trip Efficiency to re-scale the weighted_mean_installed_capacity
#     # For Paper: This proxy method assumes round-trip efficiency is temporally uniform. While this neglects the fact that losses apply only during specific charge/discharge periods, it significantly simplifies the formulation and allows for efficient integration into TSA methods. Future work could explore hybrid analytical-numerical methods for post-clustering correction.
#     weighted_mean_installed_capacity = weighted_mean_installed_capacity / (storage_process_losses['charging_efficiency']*storage_process_losses['discharging_efficiency'])

#     #apply a correction for floating point noise i.e. we want a value of 0 at the end not #1e-9
#     numeric_columns = df_timeseries.select_dtypes(include=[np.number]).columns #gather all the numeric columns including the ones generated by this function to apply correction
#     arr = df_timeseries[numeric_columns].to_numpy()
#     arr[np.isclose(arr, 0, atol=1e-9)] = 0 #threshold set to 1e-9. Could be tighter but this seems adequate for a proxy.
#     df_timeseries[numeric_columns] = arr 

#     #corrections for comparison with real SoC, translate graph such that storage is never below 0
#     df_timeseries['soc_proxy'] = df_timeseries['soc_proxy']+np.abs(np.min(df_timeseries['soc_proxy']))
   
#     return df_timeseries, weighted_mean_capacity_factors, weighted_mean_installed_capacity

def apply_temporal_rte_to_soc(df_timeseries: pd.DataFrame, surplus_col: str, charging_eff: float, discharging_eff: float):
    """
    Applies temporal charging and discharging efficiency to the surplus time series
    and returns an adjusted SoC proxy.
    """
    surplus = df_timeseries[surplus_col].values
    soc_proxy = np.zeros_like(surplus)
    
    for t in range(1, len(surplus)):
        if surplus[t] >= 0:
            soc_proxy[t] = soc_proxy[t-1] + surplus[t] * charging_eff
        else:
            soc_proxy[t] = soc_proxy[t-1] + surplus[t] / discharging_eff

    # Ensure cyclic condition: end = start
    soc_proxy -= soc_proxy[-1] * np.linspace(0, 1, len(soc_proxy))
    
    return pd.Series(soc_proxy, index=df_timeseries.index)

def decompose_surplus(
    df_timeseries: pd.DataFrame,
    method: str = 'gaussian',
    time_horizon_hours: int = 24,
    timestamp_col: str = None,
    charging_efficiency: float = 1.0,
    discharging_efficiency: float = 1.0
) -> pd.DataFrame:
    """
    Decomposes surplus into SDES (short-duration) and LDES (long-duration) components,
    and applies temporal RTE to resulting SoC proxies.
    """
    if 'surplus' not in df_timeseries.columns:
        raise ValueError("DataFrame must contain a 'surplus' column.")

    # Extract timestamps
    if timestamp_col:
        timestamps = pd.to_datetime(df_timeseries[timestamp_col])
    else:
        timestamps = pd.to_datetime(df_timeseries.index)

    surplus = df_timeseries['surplus'].values
    n = len(surplus)

    # Calculate timestep in hours
    timestep_hours = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 3600
    samples_per_window = int(round(time_horizon_hours / timestep_hours))
    if samples_per_window % 2 == 0:
        samples_per_window += 1
    half_window = samples_per_window // 2

    if method == 'fft_lowpass':
        # FFT-based low-pass filter
        freqs = fftfreq(n, d=timestep_hours)
        surplus_fft = fft(surplus)
        cutoff_freq = 1 / time_horizon_hours
        mask = np.abs(freqs) <= cutoff_freq
        filtered_fft = surplus_fft * mask
        surplus_LDES = np.real(ifft(filtered_fft))
        surplus_SDES = surplus - surplus_LDES
    else:
        # Time-domain smoothing kernels
        if method == 'moving_average':
            kernel = np.ones(samples_per_window) / samples_per_window
        elif method == 'triangular':
            kernel = np.array([1 - abs(i - half_window)/half_window for i in range(samples_per_window)])
            kernel /= kernel.sum()
        elif method == 'gaussian':
            sigma = samples_per_window / 6
            x = np.arange(samples_per_window) - half_window
            kernel = np.exp(-0.5 * (x / sigma) ** 2)
            kernel /= kernel.sum()
        else:
            raise ValueError("Unsupported method. Choose from 'moving_average', 'triangular', 'gaussian', or 'fft_lowpass'.")

        surplus_LDES = convolve(surplus, kernel, mode='same')
        surplus_SDES = surplus - surplus_LDES

    # Apply temporal SoC logic
    soc_proxy_LDES = apply_temporal_rte_to_soc(df_timeseries.assign(temp_surplus=surplus_LDES), 'temp_surplus', charging_efficiency, discharging_efficiency)
    soc_proxy_SDES = apply_temporal_rte_to_soc(df_timeseries.assign(temp_surplus=surplus_SDES), 'temp_surplus', charging_efficiency, discharging_efficiency)

    df_result = df_timeseries.copy()
    df_result['surplus_LDES'] = surplus_LDES
    df_result['surplus_SDES'] = surplus_SDES
    df_result['soc_proxy_LDES'] = soc_proxy_LDES
    df_result['soc_proxy_SDES'] = soc_proxy_SDES

    return df_result

def generate_soc_proxy_3(
    df_timeseries: pd.DataFrame, 
    demand_field: str='demand_power', 
    renewables_fields_and_weights: dict = {
        'solar':1,
        'onshore_wind': 1,
        'offshore_wind': 1
        },
    dispatchable_techs: dict = {
        'known_dispatchable_capacity_portion_mean_demand': .25 #we know that 3.3GW nuclear makes up c.25% of 13GW mean hourly demand with a high uptime
        # 'relative_dispatchable': 0 #TODO: add in capacity for relative dispatchable
    },
    storage_process_losses: dict = {
    'charging_efficiency': 0.65 * 0.99, #electrolyser efficiency * ldes injection efficiency
    'discharging_efficiency': 0.56 * 0.99 #electrolyser efficiency * ldes injection efficiency
    }
    ):

    #Quick process for checking all the fields exist and also gathering the names of the renewables
    missing_fields = []
    renewable_fields = []
    if  demand_field not in df_timeseries.columns:
        missing_fields.append(demand_field)
    for key in renewables_fields_and_weights:
        renewable_fields.append(key) 
        if key not in df_timeseries:
            missing_fields.append(key)
    if missing_fields:
        raise Exception(f"The following fields are missing from the dataframe: {missing_fields}")

    # Adjust demand for known dispatchables. Subtract from demand that which is expected to be fulfilled regularly via dispatchable clean technologies e.g. nuclear, hydro
    df_timeseries[demand_field] = df_timeseries[demand_field] - dispatchable_techs['known_dispatchable_capacity_portion_mean_demand']*np.mean(df_timeseries[demand_field])

    #normalise capacity factors by sum, for unity across summation
    renewables_fields_and_weights, _ = normalise_by_method(renewables_fields_and_weights, method='sum')

    #compute the weighted mean capacity factor for each technology and the combined value
    weighted_mean_capacity_factors = {}
    for tech in renewable_fields:
        weighted_mean_capacity_factors[tech] = df_timeseries[tech].mean()*renewables_fields_and_weights[tech]
    weighted_mean_capacity_factors['combined'] = sum(weighted_mean_capacity_factors.values())
    #and the weighted series of capacity factors
    df_timeseries['mean_capacity_factor'] = df_timeseries[list(renewables_fields_and_weights)].mul(pd.Series(renewables_fields_and_weights)).sum(axis=1)

    #compute mean installed capacity
    weighted_mean_installed_capacity = np.sum(df_timeseries[demand_field]) / np.sum(df_timeseries['mean_capacity_factor'])

    #compute total surplus
    df_timeseries['surplus'] = df_timeseries['mean_capacity_factor']*weighted_mean_installed_capacity-df_timeseries[demand_field]

    #Decompose the total surplus into LDES vs SDES components i.e. salt cavern storgage versus battery
    # LDES does not deliver all the timeshifting of energy, we apply a smoothing function on a rolling window to extract the components that factor into LDES SoC versus alternative SDES operations

    df_timeseries = decompose_surplus(
        df_timeseries,
        method='fft_lowpass',
        time_horizon_hours=24,
        timestamp_col='timesteps',
        charging_efficiency=storage_process_losses['charging_efficiency'],
        discharging_efficiency=storage_process_losses['discharging_efficiency']
    )
    
    #correction for technology losses:
    # !LIMITATION! Ideally, we would factor these efficiencies into the surplus calculations, but this breaks the linearity of the problem and it can no longer be solved analytically, requiring a numerical method
    # This defeats the goal of the proxy which is to be lightweight. Here we use Round Trip Efficiency to re-scale the weighted_mean_installed_capacity
    # For Paper: This proxy method assumes round-trip efficiency is temporally uniform. While this neglects the fact that losses apply only during specific charge/discharge periods, it significantly simplifies the formulation and allows for efficient integration into TSA methods. Future work could explore hybrid analytical-numerical methods for post-clustering correction.
    # weighted_mean_installed_capacity = weighted_mean_installed_capacity / (storage_process_losses['charging_efficiency']*storage_process_losses['discharging_efficiency'])

    #apply a correction for floating point noise i.e. we want a value of 0 at the end not #1e-9
    numeric_columns = df_timeseries.select_dtypes(include=[np.number]).columns #gather all the numeric columns including the ones generated by this function to apply correction
    arr = df_timeseries[numeric_columns].to_numpy()
    arr[np.isclose(arr, 0, atol=1e-9)] = 0 #threshold set to 1e-9. Could be tighter but this seems adequate for a proxy.
    df_timeseries[numeric_columns] = arr 

    #corrections for comparison with real SoC, translate graph such that storage is never below 0
    df_timeseries['soc_proxy_LDES'] = df_timeseries['soc_proxy_LDES']+np.abs(np.min(df_timeseries['soc_proxy_LDES']))
    df_timeseries['soc_proxy_SDES'] = df_timeseries['soc_proxy_SDES']+np.abs(np.min(df_timeseries['soc_proxy_SDES']))
   
    return df_timeseries, weighted_mean_capacity_factors, weighted_mean_installed_capacity


#  ------------------------------------------------------------------
# 
#                       SoC Assessment metrics
# 
# ------------------------------------------------------------------

def first_derivative_correlation(y_true, y_pred, standardise=False):
    """
    Computes the Pearson correlation between the first derivatives of two time series.
    
    Parameters:
    - y_true: pd.Series — the reference time series
    - y_pred: pd.Series — the predicted time series
    - standardise: bool — whether to standardise the differences before correlation

    Returns:
    - float — Pearson correlation of the first derivatives
    """
    # Compute first differences
    y_true_diff = y_true.diff().dropna()
    y_pred_diff = y_pred.diff().dropna()
    
    # Align lengths
    min_len = min(len(y_true_diff), len(y_pred_diff))
    y_true_diff = y_true_diff.iloc[:min_len]
    y_pred_diff = y_pred_diff.iloc[:min_len]

    # Optionally standardise
    if standardise:
        y_true_diff = (y_true_diff - y_true_diff.mean()) / y_true_diff.std()
        y_pred_diff = (y_pred_diff - y_pred_diff.mean()) / y_pred_diff.std()
    
    # Compute Pearson correlation
    corr, _ = pearsonr(y_true_diff, y_pred_diff)
    return corr

def peak_timing_error(y_true, y_pred, field_reference: str = 'index'):

     #get id of prediction peak,
    id_peak_true = y_true.idxmax() #get id of true profile peak, representing the hour
    id_peak_pred = y_pred.idxmax()
    
    #if index is to be used for hour in period
    if field_reference == 'index':
        peak_true = id_peak_true
        peak_pred = id_peak_pred
        series_max = y_true.index.max()
        series_min = y_true.index.min()

    #if another column is to be used for hour in period
    else:
        peak_true = y_true.loc[field_reference,id_peak_true]
        peak_pred = y_pred.loc[field_reference,id_peak_pred]
        series_max = y_true[field_reference].max()
        series_min = y_true[field_reference].min()

    if isinstance(peak_true,Timestamp) and isinstance(peak_pred,Timestamp):
        diff_in_hours = (peak_true-peak_pred).total_seconds() / 3600
        max_span_hours = (series_max - series_min).total_seconds() / 3600
        result = diff_in_hours / max_span_hours
    else:
        result = (peak_true-peak_pred)/(series_max-series_min)
    
    return result #calcualte difference and divide by highest id i.e. divide by hours in the model e.g. 8760 for a 1 year model

def pearson_r_standardised(y_true, y_pred):

    #normalise with respect to standard dev
    ref_stardardised = (y_true-np.mean(y_true)) / np.std(y_true)
    proxy_stardardised = (y_pred-np.mean(y_pred)) / np.std(y_pred)

    #calculate pearson r coeff
    r, _ = pearsonr(ref_stardardised, proxy_stardardised)

    return r

def cos_similarity(y_true, y_pred):
    return cosine_similarity([y_true], [y_pred])[0][0]

def standardised_profile_comparison(y_true,y_pred):

    e_peak_time = peak_timing_error(y_true, y_pred,)
    e_pearson_r = pearson_r_standardised(y_true, y_pred)
    e_cos_sim = cos_similarity(y_true, y_pred)
    e_fd_corr = first_derivative_correlation(y_true, y_pred, standardise=True)

    return e_peak_time,e_pearson_r,e_cos_sim, e_fd_corr
