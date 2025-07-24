import pandas as pd
import numpy as np
from scipy.signal import convolve
from numpy.fft import fft, ifft, fftfreq

def compute_soc_proxy(surplus, charging_eff, discharging_eff):
    soc_proxy = np.zeros_like(surplus)
    for t in range(1, len(surplus)):
        if surplus[t] >= 0:
            soc_proxy[t] = soc_proxy[t-1] + surplus[t] * charging_eff
        else:
            soc_proxy[t] = soc_proxy[t-1] + surplus[t] / discharging_eff
    soc_proxy -= soc_proxy[-1] * np.linspace(0, 1, len(soc_proxy))
    return soc_proxy

def apply_temporal_rte_to_soc(df: pd.DataFrame, surplus_col: str, charging_eff: float, discharging_eff: float):
    soc_proxy = compute_soc_proxy(df[surplus_col].values.astype(np.float64), charging_eff, discharging_eff)
    return pd.Series(soc_proxy, index=df.index)

def decompose_surplus(
    df: pd.DataFrame,
    soc_decomposition_method: str = 'gaussian',
    time_horizon_hours: int = 24,
    timestamp_col: str = None,
    charging_efficiency: float = 1.0,
    discharging_efficiency: float = 1.0
):
    
    if 'surplus' not in df.columns:
        raise ValueError("DataFrame must contain a 'surplus' column.")

    timestamps = pd.to_datetime(df[timestamp_col]) if timestamp_col else pd.to_datetime(df.index)
    surplus = df['surplus'].values
    n = len(surplus)
    timestep_hours = (timestamps.iloc[1] - timestamps.iloc[0]).total_seconds() / 3600

    samples = int(round(time_horizon_hours / timestep_hours)) | 1  # force odd number

    if soc_decomposition_method == 'fft_lowpass':
        freqs = fftfreq(n, d=timestep_hours)
        surplus_fft = fft(surplus)
        mask = np.abs(freqs) <= (1 / time_horizon_hours)
        surplus_LDES = np.real(ifft(surplus_fft * mask))
    else:
        half_window = samples // 2
        if soc_decomposition_method == 'moving_average':
            kernel = np.ones(samples) / samples
        elif soc_decomposition_method == 'triangular':
            kernel = np.array([1 - abs(i - half_window)/half_window for i in range(samples)])
            kernel /= kernel.sum()
        elif soc_decomposition_method == 'gaussian':
            sigma = samples / 6
            x = np.arange(samples) - half_window
            kernel = np.exp(-0.5 * (x / sigma)**2)
            kernel /= kernel.sum()
        else:
            raise ValueError(f"Unsupported method: {soc_decomposition_method}")
        surplus_LDES = convolve(surplus, kernel, mode='same')

    surplus_SDES = surplus - surplus_LDES

    df['surplus_LDES'] = surplus_LDES
    df['surplus_SDES'] = surplus_SDES
    df['soc_proxy_LDES'] = apply_temporal_rte_to_soc(df.assign(temp_surplus=surplus_LDES), 'temp_surplus', charging_efficiency, discharging_efficiency)
    df['soc_proxy_SDES'] = apply_temporal_rte_to_soc(df.assign(temp_surplus=surplus_SDES), 'temp_surplus', charging_efficiency, discharging_efficiency)

    return df

def generate_soc_proxy(
    df: pd.DataFrame, 
    demand_field: str = 'demand_power',
    renewables_fields_and_weights: dict = {'solar': 1, 'onshore_wind': 1, 'offshore_wind': 1},
    dispatchable_techs: dict = {'known_dispatchable_capacity_portion_mean_demand': 0.25},
    storage_process_losses: dict = {
        'charging_efficiency': 0.65 * 0.99,
        'discharging_efficiency': 0.56 * 0.99
    },
    soc_decomposition: dict = {
        'method': 'fft_lowpass',
        'time_horizon_hours': 24
    },
    timestamp_col: str = 'timesteps'
):
    
    supported_methods = {'fft_lowpass', 'gaussian', 'triangular', 'moving_average'}
    if soc_decomposition['method'] not in supported_methods:
        raise ValueError(f"Method '{soc_decomposition['method']}' not supported. Choose from: {supported_methods}")

    renewable_fields = list(renewables_fields_and_weights.keys())
    weights = np.array(list(renewables_fields_and_weights.values()), dtype=float)
    weights /= weights.sum()

    # Validate required fields
    required_fields = [demand_field] + renewable_fields
    missing = [col for col in required_fields if col not in df.columns]
    if missing:
        raise ValueError(f"Missing fields in dataframe: {missing}")
    
    # Subtract base dispatchable capacity
    df[demand_field] -= dispatchable_techs.get('known_dispatchable_capacity_portion_mean_demand', 0) * df[demand_field].mean()

    # Raw mean capacity factor per technology (unweighted)
    capacity_factors = {
        tech: df[tech].mean()
        for tech in renewable_fields
    }

    capacity_factors['weighted_mean'] = sum(
        weights[i] * capacity_factors[tech]
        for i, tech in enumerate(renewable_fields)
    )

    df['mean_capacity_factor'] = df[renewable_fields].mul(weights, axis=1).sum(axis=1)
    weighted_installed_capacity = df[demand_field].sum() / df['mean_capacity_factor'].sum()
    df['surplus'] = df['mean_capacity_factor'] * weighted_installed_capacity - df[demand_field]

    df = decompose_surplus(
        df,
        soc_decomposition_method=soc_decomposition['method'],  # <-- Use method argument
        time_horizon_hours=soc_decomposition['time_horizon_hours'],
        timestamp_col=timestamp_col,
        charging_efficiency=storage_process_losses['charging_efficiency'],
        discharging_efficiency=storage_process_losses['discharging_efficiency']
    )

    # Clip near-zero floating noise
    for col in ['soc_proxy_LDES', 'soc_proxy_SDES', 'surplus_LDES', 'surplus_SDES']:
        df[col] = np.where(np.abs(df[col]) < 1e-9, 0, df[col])

    # Shift proxies so minimum value is 0
    df['soc_proxy_LDES'] -= df['soc_proxy_LDES'].min()
    df['soc_proxy_SDES'] -= df['soc_proxy_SDES'].min()

    # Weighted installed capacities
    installed_caps_nominal = {
    tech: weights[i] * weighted_installed_capacity
    for i, tech in enumerate(renewable_fields)
    }
    installed_caps_nominal['total'] = weighted_installed_capacity

    return df, capacity_factors, installed_caps_nominal
