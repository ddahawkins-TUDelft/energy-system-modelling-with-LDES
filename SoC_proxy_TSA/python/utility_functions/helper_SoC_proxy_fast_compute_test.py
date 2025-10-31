import pandas as pd
import numpy as np
import time
from scipy.signal import convolve
from numpy.fft import fft, ifft, fftfreq

def compute_soc_proxy(surplus, charging_eff, discharging_eff):
    """
    Returns surplus_corr and soc_proxy_corr (cumsum) with an end–middle–end cosine correction.
    - frac    : fraction of the series length used at EACH end for the taper (0<frac<=0.5)
    - cushion : >=0. Adds some correction to BOTH ends (so the end window isn't zeroed).
                cushion=0 uses only the start lobe; larger values put some mass on the end too.
    """

    soc_proxy = np.cumsum(surplus)

    # mag_mismatch = (soc_proxy[0] - soc_proxy[-1])*0.5

    # soc_proxy -= mag_mismatch
    
    # FRAC = 0.01
    # n = soc_proxy.size

    # # length of each end taper
    # L = max(1, min(n // 2, int(round(FRAC * n))))

    # # half-Hann (cosine) window
    # t = np.arange(L, dtype=float)
    # l_linspace = np.linspace(0,1, L)
    # r_linspace = np.linspace(1,0, L)
    # if FRAC<0.5:
    #     scalar_vector= np.concatenate((l_linspace, np.ones(n-2*L), r_linspace)) 
    # else:
    #     scalar_vector= np.concatenate((l_linspace, r_linspace)) 

    # soc_proxy*= scalar_vector
    # soc_proxy+= mag_mismatch

    return soc_proxy

def apply_temporal_rte_to_soc(df: pd.DataFrame, surplus_col: str, charging_eff: float, discharging_eff: float, return_corrected_surplus: bool = True, check_cyclical: bool = False):
    """
    Wrapper to apply compute_soc_proxy to a DataFrame column and return Series.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing the surplus column.
        surplus_col (str): Column name containing surplus values.
        charging_eff (float): Charging efficiency.
        discharging_eff (float): Discharging efficiency. 
        return_corrected_surplus(bool): decision to also return the corrected surpluses. Defaults true.
        check_cyclical (bool): check that surpluses sum to 0 (no energy lost or created). Defaults false.

    Returns:
        pd.Series: Computed SoC proxy.
        (Optional) pd.Series: corrected_surplus
    """

    soc_proxy = compute_soc_proxy(df[surplus_col].values.astype(np.float64), charging_eff, discharging_eff)
    soc_series = pd.Series(soc_proxy, index=df.index)

    if not return_corrected_surplus and not check_cyclical:
        return soc_series

    # Compute delta SoC
    delta_soc = soc_series.diff().fillna(0)

    #   Corrected surpluses is an abstract represention of the fluctuations without efficiency losses/gains
    #   For this reason, surpluses do not sum to 0 and this signal should be avoided unless properly understood
    # Infer actual surplus from SoC deltas and efficiencies
    corrected_surplus = np.where(
        delta_soc >= 0,
        delta_soc / charging_eff,
        delta_soc * discharging_eff
    )

    df_surpluses = pd.Series(delta_soc, index=df.index)

    if check_cyclical:
        residual = soc_series.iloc[-1] - soc_series.iloc[0]
        if abs(residual) > 1e-6:
            print(f"⚠ Warning: SoC not cyclical (end-start = {residual:.3e})")
        

    if return_corrected_surplus:
        return soc_series, df_surpluses
    else:
        return soc_series

def decompose_surplus(
    df: pd.DataFrame,
    soc_decomposition_method: str = 'gaussian',
    time_horizon_hours: int = 24,
    timestamp_col: str = None,
    charging_efficiency: float = 1.0,
    discharging_efficiency: float = 1.0
):
    """
    Decomposes surplus signal into short- and long-duration components (SDES and LDES),
    and computes SoC proxies for each using round-trip efficiencies.

    Parameters:
        df (pd.DataFrame): Input DataFrame with a 'surplus' column.
        soc_decomposition_method (str): Method for smoothing ('gaussian', 'moving_average', 'triangular', 'fft_lowpass').
        time_horizon_hours (int): Timescale threshold separating LDES from SDES.
        timestamp_col (str): Optional name of timestamp column. If None, uses index.
        charging_efficiency (float): Charging efficiency for SoC calculation.
        discharging_efficiency (float): Discharging efficiency for SoC calculation.

    Returns:
        pd.DataFrame: Input DataFrame augmented with:
            - surplus_LDES / surplus_SDES
            - soc_proxy_LDES / soc_proxy_SDES
    """
    if 'surplus' not in df.columns:
        raise ValueError("DataFrame must contain a 'surplus' column.")
    
    # Get timestamps safely
    if timestamp_col:
        timestamps = pd.to_datetime(df[timestamp_col])
    else:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Timestamp column not provided and index is not a DatetimeIndex.")
        timestamps = df.index.to_series()

    # Ensure timestamps are sorted
    timestamps = timestamps.sort_values()

    # Validate there are at least 2 unique time steps
    if timestamps.nunique() < 2:
        raise ValueError("Not enough unique timestamps to compute time intervals.")

    timestep_seconds = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds()
    if timestep_seconds == 0:
        raise ValueError("Timestamps have zero time difference. Check the index or timestamp column.")
    timestep_hours = timestep_seconds / 3600

    surplus = df['surplus'].values
    n = len(surplus)
    timestep_hours = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 3600

    # Compute number of samples equivalent to the smoothing horizon
    samples = int(round(time_horizon_hours / timestep_hours)) | 1  # Ensure it's odd

    if soc_decomposition_method == 'fft_lowpass':
        # FFT-based low-pass filter: zero out high frequencies
        freqs = fftfreq(n, d=timestep_hours)
        surplus_fft = fft(surplus)
        mask = np.abs(freqs) <= (1 / time_horizon_hours)
        surplus_LDES = np.real(ifft(surplus_fft * mask))
    else:
        # Construct time-domain smoothing kernel
        half_window = samples // 2
        if soc_decomposition_method == 'moving_average':
            kernel = np.ones(samples) / samples
        elif soc_decomposition_method == 'triangular':
            kernel = np.array([1 - abs(i - half_window) / half_window for i in range(samples)])
            kernel /= kernel.sum()
        elif soc_decomposition_method == 'gaussian':
            sigma = samples / 6  # covers ~99% in ±3σ
            x = np.arange(samples) - half_window
            kernel = np.exp(-0.5 * (x / sigma)**2)
            kernel /= kernel.sum()
        else:
            raise ValueError(f"Unsupported method: {soc_decomposition_method}")
        surplus_LDES = convolve(surplus, kernel, mode='same')
    

    # SDES = Original - LDES
    surplus_SDES = surplus - surplus_LDES

    # Store results in dataframe
    df['surplus_LDES'] = surplus_LDES
    df['surplus_SDES'] = surplus_SDES

    df['soc_proxy_LDES'], df['surplus_LDES'] = apply_temporal_rte_to_soc(df.assign(temp_surplus=surplus_LDES), 'temp_surplus', charging_efficiency, discharging_efficiency, return_corrected_surplus=True, check_cyclical=True)
    df['soc_proxy_SDES'], df['surplus_SDES'] = apply_temporal_rte_to_soc(df.assign(temp_surplus=surplus_SDES), 'temp_surplus', charging_efficiency, discharging_efficiency, return_corrected_surplus=True, check_cyclical=True)

    # print('>>> Proxy successfully applied, including a Round Trip Efficiency correction and a check of surplus balances.')

    return df

def generate_soc_proxy(
    df: pd.DataFrame, 
    demand_field: str = 'demand_power',
    renewables_fields_and_weights: dict = {'solar': 1, 'onshore_wind': 1, 'offshore_wind': 1},
    dispatchable_techs: dict = {'known_dispatchable_capacity': 3300},
    storage_process_losses: dict = {
        'charging_efficiency': 0.65 * 0.99,
        'discharging_efficiency': 0.56 * 0.99
    },
    soc_decomposition: dict = {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    },
    timestamp_col: str = None
):
    """
    Function that takes in timeseries data and configution options, and returns state of charge (SoC) proxies for both long-duration energys storages and short-duration energy storages. 
    This proxy scales with demand and considers the time-varying nature of renewable capacity factors. It assumes a cyclical condition whereby SoC is equal at the beginning and end of
    the time period i.e. no net change across domain.

    Parameters:
        df (pd.DataFrame): Input time series with demand and renewable capacity factors.
        demand_field (str): Column name for electricity demand (power units).
        renewables_fields_and_weights (dict): Technology-to-weight mapping (e.g. {'solar': 1.0, 'wind': 2.0}).
        dispatchable_techs (dict): Assumed dispatchable supply as portion of average demand. e.g. if average demand is 10GWh at each hour and nuclear capacity is 3GW, the corresponding parameter can be set to 0.3
        storage_process_losses (dict): Charging and discharging efficiencies (between 0 and 1).
        soc_decomposition (dict): Method and timescale for LDES/SDES split.
        timestamp_col (str): Column to use for time information (defaults to index if None).

    Returns:
        df (pd.DataFrame): DataFrame with new columns for surplus, SoC proxies, and decomposition.
        capacity_factors (dict): Unweighted mean CFs per technology and weighted mean for mix.
        installed_caps_nominal (dict): Nominal installed capacities per tech (based on weights).
    """
    supported_methods = {'fft_lowpass', 'gaussian', 'triangular', 'moving_average'}
    if soc_decomposition['method'] not in supported_methods:
        raise ValueError(f"Method '{soc_decomposition['method']}' not supported. Choose from: {supported_methods}")

    renewable_fields = list(renewables_fields_and_weights.keys())
    weights = np.array(list(renewables_fields_and_weights.values()), dtype=float)
    weights /= weights.sum()  # Normalize weights to sum to 1

    # Validate required fields
    required_fields = [demand_field] + renewable_fields
    missing = [col for col in required_fields if col not in df.columns]
    if missing:
        raise ValueError(f"Missing fields in dataframe: {missing}")
    
    # Subtract assumed dispatchable baseline capacity from demand
    df[demand_field] -= dispatchable_techs.get('known_dispatchable_capacity', 0)

    # Raw (unweighted) average capacity factor per technology
    capacity_factors = {
        tech: df[tech].mean()
        for tech in renewable_fields
    }

    # Weighted average capacity factor for the full mix
    capacity_factors['weighted_mean'] = sum(
        weights[i] * capacity_factors[tech]
        for i, tech in enumerate(renewable_fields)
    )

    # Create time-varying mean capacity factor (blended)
    df['mean_capacity_factor'] = df[renewable_fields].mul(weights, axis=1).sum(axis=1)

    # Calculate total nominal installed capacity needed to meet demand on average
    weighted_installed_capacity = df[demand_field].sum() / df['mean_capacity_factor'].sum()

    storage_cap = 0
    debt = np.zeros(df.shape[0]) 
    debt[0] = 100000 #initial offset to trigger whileloop
    
    lim = df.shape[0]

    #loop metrics
    loop_count = 0
    t_start = time.time()

    list_curtailments = []
    list_debt_starts = []


    
    

    # --- Setup ---
    d = df[demand_field].to_numpy()
    
    cumG = np.cumsum(df['mean_capacity_factor'].to_numpy() * weighted_installed_capacity * 1)
    cumD = np.cumsum(d)
    curtailment_factor =1/(np.max(cumD[cumG>0] / cumG[cumG>0]))

    curtailment_factor = 0.75


    g_curtailed = df['mean_capacity_factor'].to_numpy() * weighted_installed_capacity / curtailment_factor
    cumG = np.cumsum(g_curtailed)
    rnd = g_curtailed-d
    cum_available_surplus = np.cumsum(np.maximum(rnd, 0)) #represents available historic energy to meet future deficits

    # --- Backward loop: reserve energy for deficits ---
    lim = len(df)

    pos = np.maximum(rnd, 0.0)                 # surplus per step
    neg = np.maximum(-rnd, 0.0)                # deficit per step

    pos_eff = pos*storage_process_losses['charging_efficiency']
    neg_eff = neg/storage_process_losses['discharging_efficiency']

    charge = np.zeros(lim)                     # energy reserved/charged at time k (LIFO source)
    discharge = np.zeros(lim)                  # your convention: negative at deficit times
    remaining = pos_eff.copy()                     # mutable buckets of available surplus

    k = lim - 1                                # pointer to latest (rightmost) surplus bucket

    for t in range(lim - 1, -1, -1):           # walk backward in time
        deficit = neg_eff[t]
        if deficit == 0.0:
            continue

        discharge[t] = -deficit                # record discharge (negative by your convention)

        # Do not borrow from the future: clamp k to t
        if k > t:
            k = t

        # Consume surplus from k..0 until we cover the deficit (LIFO)
        while deficit > 1e-12 and k >= 1e-12: #using 1e12 instead of 0 to allow for floating point errors
            # move k left to next bucket with remaining surplus
            while k >= 1e-12 and remaining[k] <= 1e-12:
                k -= 1
            if k < 1e-12:
                break

            take = remaining[k] if remaining[k] < deficit else deficit
            charge[k] += take
            remaining[k] -= take
            deficit -= take

            if remaining[k] <= 1e-12:
                k -= 1

        if remaining[0] < 0:
            print('Insufficient renewables')

    
    df['surplus'] = charge+discharge
    print('charge/discharge arrays constructed')
        # Optional: track unmet deficit if supply is insufficient
        # if deficit > 0.0:
        #     lost_load[t] = deficit


    # for i in range(0, lim):
    #     t = lim - (i + 1)
    #     if i == 0:
    #         charge[t] = 0
    #         discharge[t] = 0  # end condition for cyclicality
    #         continue

    #     surplus = rnd[t]
    #     # if surplus > 0:
    #     #     charge[t] = surplus
    #         #discharge remains 0 @ t
        
    #     if surplus < 0:
    #         #discharge
    #         discharge[t] = surplus
    #         #now look back through time, and instruct the storage to reserve energy at relevant t values (i.e. assign values for charge)
    #         rnd_threshold = cum_available_surplus[t] + surplus
            
            
    #         #charge instructions, should align with required energy for deficit at t, LIFO
    #         #get ids of events above thresholds
    #         idx = np.where(cum_available_surplus[:t+1] >= rnd_threshold)[0]
    #         #get the available energy, note we use idx-1 because with cumsum everything gets shifted
    #         charge[idx] = np.maximum(0,rnd[idx]-np.maximum(0,rnd_threshold-cum_available_surplus[idx-1]))
    #         cum_available_surplus[idx-1] -= charge[idx]


    #         # next steps:
    #         # allocate deficit across t values backwards, has to be a loop unfortunately
    #         # update the rnd and cumRND values for those timesteps so we dont borrow against the same step twice
    #         #     ...etc.
            
            


    #         cumRND[:t][cumRND[:t] >= s_required] = s_required

    #     if cumRND[t] < 0: 
    #         print('t_value', t) 
    #         print('done')
            
    # # --- Construct surplus and SoC proxy ---
    # surplus = charge - discharge
    # soc = np.cumsum(surplus)

    # Enforce cyclic SoC (SoC[0] = SoC[T])
    # soc -= np.linspace(soc[0], soc[-1], lim)


    print('time: ', time.time() -t_start, 'n_loops: ', loop_count)
    print('curtailment forecast', f"{1/curtailment_factor:.2%}")


    # df['surplus'] = np.diff(debt, prepend=debt[0])
        
    # Compute surplus as supply minus demand
    # df['surplus'] = df['mean_capacity_factor'] * weighted_installed_capacity*curtailment_factor - df[demand_field]

    # Apply decomposition and compute SoC proxies
    df = decompose_surplus(
        df,
        soc_decomposition_method=soc_decomposition['method'],
        time_horizon_hours=soc_decomposition['time_horizon_hours'],
        timestamp_col=timestamp_col,
        charging_efficiency=storage_process_losses['charging_efficiency'],
        discharging_efficiency=storage_process_losses['discharging_efficiency']
    )

    # Clean very small noise
    for col in ['soc_proxy_LDES', 'soc_proxy_SDES', 'surplus_LDES', 'surplus_SDES']:
        df[col] = np.where(np.abs(df[col]) < 1e-9, 0, df[col])

    # rescale indirectly proportionally to the RES overinvestment factor applied before
    df['soc_proxy_LDES'] *= curtailment_factor
    df['soc_proxy_SDES'] *= curtailment_factor

    # Shift proxies so they begin from 0
    df['soc_proxy_LDES'] -= df['soc_proxy_LDES'].min()
    df['soc_proxy_SDES'] -= df['soc_proxy_SDES'].min()

    # Calculate nominal installed capacity per technology (based on weights)
    installed_caps_nominal = {
        tech: weights[i] * weighted_installed_capacity
        for i, tech in enumerate(renewable_fields)
    }
    installed_caps_nominal['total'] = weighted_installed_capacity

    return df, capacity_factors, installed_caps_nominal
