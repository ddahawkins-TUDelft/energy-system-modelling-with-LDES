
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import calliope
from netCDF4 import Dataset
import  yaml

# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "axes.unicode_minus": False,
# })
# mpl.rcParams["pgf.preamble"] = r""

data = {
    'id': [
        '1dff92aac973a8452fdc',
        '1fde1fb23ae1a3833840',
        '788f5e7bbbe933d3e33a',
        '128bd3193cc9c5e26017',
        '2109cf7332b1ea3a0da4',
        'c2da63c2dcb882131614',
        ],
    'model_code': [
        'cTSA-5Y-N-30reps',
        'cTSA-5Y-EXO-2-30reps',
        'cTSA-5Y-EXO-5-30reps',
        'cTSA-5Y-EXO-10-30reps',
        'cTSA-5Y-EXO-20-30reps',
        'cTSA-5Y-EXO-50-30reps',
        ],
}
reference_model = 'SoC_proxy_TSA/data/calliope_models/standard_2015_2019_reference.nc'

colour_1 = '#0D0887'   
colour_2 = '#CC4778' 

def cem_results(df, path_reference):

    # reference
    model_reference = calliope.read_netcdf(path_reference)
    power_caps_reference, energy_caps_reference = get_capacities(model_reference)


    list_power_cap_mean_errors = []
    list_ldes_cap_error = []

    for model_id in df['id']:
        model_test = read_clustered_netcdf(f"SoC_proxy_TSA/data/calliope_models/{model_id}.nc")
        power_caps_test, energy_caps_test = get_capacities(model_test)

        #metrics
        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)
        e_storage,_ = relative_error(energy_caps_reference, energy_caps_test)

        list_power_cap_mean_errors.append(e_mean_power)
        list_ldes_cap_error.append(e_storage['h2_salt_cavern'])

    df.insert(0,'macme',list_power_cap_mean_errors)
    df.insert(0,'ldes_error',list_ldes_cap_error)
    return df

def read_clustered_netcdf(path):

    p = Path(path).resolve()
    if not p.is_file():
        raise Exception('File does not exist at: {path}')
    with Dataset(p, "a") as nc:  # append mode
        g = nc.groups["attrs"]
        cfg = yaml.safe_load(g.getncattr("config"))
        cfg.get("init", {}).pop("time_cluster", None)  # remove lingering clustering
        g.setncattr("config", yaml.safe_dump(cfg))

    return calliope.read_netcdf(path)

def get_capacities(m: calliope.Model):

    df_power_caps = (
        m.results['flow_cap']
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame('capacity')
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
        )
    
    mask_drop = (
        df_power_caps["techs"].isin(["battery", "h2_salt_cavern", "demand"])
        | df_power_caps["techs"].str.startswith("demand")
        )
    df_power_caps = df_power_caps[~mask_drop & (
            ((df_power_caps["techs"] == "electrolyser") & (df_power_caps["carriers"] == "hydrogen")) |
            ((df_power_caps["techs"] != "electrolyser") & (df_power_caps["carriers"] == "power"))
        )]
    df_power_caps.set_index("techs", inplace=True)
    
    df_energy_caps = (
        m.results['storage_cap']
        .fillna(0)
        .to_series()
        .dropna()
        .to_frame('capacity')
        .reset_index()
        .drop(columns=["nodes"], errors="ignore")
        )
    df_energy_caps = df_energy_caps[df_energy_caps['capacity']>0]
    df_energy_caps.set_index("techs", inplace=True)
    
    return df_power_caps['capacity'],df_energy_caps['capacity']

def relative_error(df_ref, df_test):

    e = (df_ref - df_test) / df_ref
    e_mean_abs = np.mean(np.abs(e))

    return e, e_mean_abs


figure_save_path = 'CEM vs W.pdf'

df = pd.DataFrame(data)

df = cem_results(df,reference_model)




fig = plt.figure(figsize=(4, 4))  # wide × short
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)

# Reference model (same color; solid = actual, dashed = proxy)
ax.scatter(df['model_code'], df['macme'], label='$\overline{\epsilon^C}$', color=colour_1, linewidth=1.2)
ax.scatter(df['model_code'], df['ldes_error'], label='$\epsilon^C_\mathrm{LDES}$', color=colour_2, linewidth=1.2)


# # Clustered model (same color; solid = actual, dashed = proxy)
# ax.plot(soc_clu_actual_d.index, soc_clu_actual_d.values,
#         label='Clustered: SoC (CEM)', color=colour_clu, linewidth=1.2)
# ax.plot(soc_clu_proxy_d .index, soc_clu_proxy_d .values,
#         label='Clustered: SoC Proxy', color=colour_clu, linewidth=1.2, linestyle='dotted')

ax.set_ylabel('Error')
ax.set_xlabel('Weighting')

# vertical grid
ax.xaxis.grid(True, which='major', linestyle=':', alpha=0.6)

ax.legend(loc='best', frameon=False)
fig.tight_layout()

# ------------------------------
# Save
# ------------------------------
# plt.savefig(figure_save_path, bbox_inches='tight')  # quick preview
plt.show()
