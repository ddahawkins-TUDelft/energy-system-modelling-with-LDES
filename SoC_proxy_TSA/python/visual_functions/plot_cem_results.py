
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
import calliope
from netCDF4 import Dataset
import  yaml
import re

# mpl.rcParams.update({
#     "text.usetex": True,
#     "pgf.texsystem": "pdflatex",
#     "pgf.rcfonts": False,
#     "axes.unicode_minus": False,
# })
# mpl.rcParams["pgf.preamble"] = r""

#WEIGHTS
# data = {
#     'id': [
#         '2109cf7332b1ea3a0da4',
#         'bbf3cd4af8938c879500',
#         'eae28782795d48106d25',
#         '1fde1fb23ae1a3833840',
#         '788f5e7bbbe933d3e33a',
#         '128bd3193cc9c5e26017',
#         'c2da63c2dcb882131614',
#         '1dff92aac973a8452fdc',
#         ],
#     'model_code': [
#         'cTSA-5Y-N-30reps',
#         'cTSA-5Y-EXO-1-30reps',
#         'cTSA-5Y-EXO-2-30reps',
#         'cTSA-5Y-EXO-5-30reps',
#         'cTSA-5Y-EXO-10-30reps',
#         'cTSA-5Y-EXO-20-30reps',
#         'cTSA-5Y-EXO-50-30reps',
#         'cTSA-5Y-EXO-100-30reps',
#         ],
#     'x_axis': [
#         'No Proxy \\ ($W_x=0$)',
#         '1',
#         '2',
#         '5',
#         '10',
#         '20',
#         '50',
#         '100',
#         ],
# }

#REP PERIODS
# data1 = {
#     'id': [
#         '71f6bd7158cb670f2e0d',
#         '1ab7b1245bf0459d20e3',
#         'b5d0294cc4725e0b5fb3',
#         '2109cf7332b1ea3a0da4',
#         'a4bc361bbdd602d1f8bf',
#         'cf9e8507a51148b889f5'
#         ],
#     # 'model_code': [  ],
#     'x_axis': [
#         '7',
#         '14',
#         '21',
#         '30',
#         '90',
#         '180'
#         ],
# }

# data2 = {
#     'id': [
#         'b437c28cf906b3e0e339',
#         'e3ee4018d873017ef7f3',
#         'e0b396f4839af82bf3ac',
#         '1dff92aac973a8452fdc',
#         '06711f9352a6f7bf1512',
#         'c396ef70d36ebf8ce5ba'
#         ],
#     # 'model_code': [  ],
#     'x_axis': [
#         '7',
#         '14',
#         '21',
#         '30',
#         '90',
#         '180'
#         ],
# }

#HORIZON
data_W0 = {
    'id': [
        '335a40cd19b374f1ed6f',
        '0f49dc7c37f9d646212d',
        '83f3c65716fa5b47d9f9',
        '41de038bec87dd0d8e84',
        '92b7061598c3f69bb108',
        'e4417bf9d379a3e356e2',
        '8ea80cd72bb2392d18d9',
        '2b88b9fbb4fea53c79eb',
        '565f355a199f1bb5c2f5',
        'f1e8140151977e99b1fe',
        'fa9a32277b03eabfac1d',
        '36dc45ae7f5c93877888',
        '40a83e7bedc42a78f160',
        'ced28461d126ce84e4eb',
        '6e89773e98ee102ab1fa',
        '2109cf7332b1ea3a0da4'
        ],
    # 'model_code': [  ],
    'x_axis': [
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '5',
        '5',
        '5',
        '5',
        '5',
        '5',
        '10'
        ],
}

data_W100 = {
    'id': [
        'a0bcf289ef6025755e82',
        'd5fd3fe4ee6854351e22',
        '474dde81a56edeec554d',
        '4493f9e8b6bd2cfef43e',
        'f0cc18e55ec12ea8423d',
        '1f5fce2c65a9d1e0b441',
        'ea32ba119af0c67f0759',
        '7c70d7a35809cf98d19a',
        '11ff6bf669c7a5d7f31d',
        '517ea7feadacf73d1f1d',
        '4f0b0824a2e3dc838304',
        'e2234a54147b5968c63f',
        '31c2f6660b3e828eb565',
        '903b15aa6bc4d989b145',
        '823ac03029d0c3b8ad59',
        '1dff92aac973a8452fdc'
        ],
    # 'model_code': [  ],
    'x_axis': [
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '2',
        '5',
        '5',
        '5',
        '5',
        '5',
        '5',
        '10'
        ],
}

config_src = pd.read_csv('SoC_proxy_TSA/data/notes/log_10_yr_WandKTests.csv')
config_src['number_reps'] = (
    config_src["model_name"]
    .str.extract(r"reps\s*=\s*(\d+)", expand=False)
    .astype("Int64")   # nullable integer dtype
)
config_src["W_proxy"] = (
    config_src["model_name"]
    .str.extract(r"W_proxy\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", expand=False)
    .astype(float)
)
tag = config_src["tvp"].str.extract(r"(?i)(shuffle[^/]*?)(?=\.csv\b)", expand=False)
ref_tag = tag.fillna("standard_2010_2019_reference")
config_src["reference_path"] = "SoC_proxy_TSA/data/calliope_models/" + ref_tag + ".nc"

df = config_src[['id','W_proxy','number_reps','reference_path']]

#filter and control plot
df=df[df['number_reps']==90]
df['x_axis'] = df['W_proxy']

# reference_model = 'SoC_proxy_TSA/data/calliope_models/standard_2010_2019_reference.nc'

colour_1 = '#0D0887'   
colour_2 = '#CC4778' 

def cem_results(df):

    # reference
    


    list_power_cap_mean_errors = []
    list_ldes_cap_error = []

    for model in df.itertuples(index=True):
        print(f'Extracting info for {model.id}')

        model_reference = calliope.read_netcdf(model.reference_path)
        power_caps_reference, energy_caps_reference = get_capacities(model_reference)
        
        model_test = read_clustered_netcdf(f"SoC_proxy_TSA/data/calliope_models/{model.id}.nc")
        power_caps_test, energy_caps_test = get_capacities(model_test)

        #metrics
        _, e_mean_power = relative_error(power_caps_reference, power_caps_test)
        e_storage,_ = relative_error(energy_caps_reference, energy_caps_test)

        list_power_cap_mean_errors.append(e_mean_power)
        list_ldes_cap_error.append(np.abs(e_storage['h2_salt_cavern']))

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




df = cem_results(df)
# df2 = cem_results(df2,reference_model)


# 1) mask: True = shuffled/artificial, False = real
mask_shuffle = df["reference_path"].str.contains(r"shuffle", case=False, na=False)

# 2) split views
real    = df[~mask_shuffle]
shuf    = df[mask_shuffle]

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(1, 1, 1)
ax.set_axisbelow(True)

# --- macme series (colour_1) ---
# real (filled)
ax.scatter(real["x_axis"], real["macme"],
           label=r'$\overline{\epsilon^C}$',
           color=colour_1, linewidth=1.2)
# shuffled (hollow)
ax.scatter(shuf["x_axis"], shuf["macme"],
           label="_nolegend_",                 # avoid duplicate legend entry
           edgecolors=colour_1, facecolors='none', linewidth=1.2)

# --- LDES series (colour_2) ---
# real (filled)
ax.scatter(real["x_axis"], real["ldes_error"],
           label=r'$\epsilon^C_{\mathrm{LDES}}$',
           color=colour_2, linewidth=1.2)
# shuffled (hollow)
ax.scatter(shuf["x_axis"], shuf["ldes_error"],
           label="_nolegend_",
           edgecolors=colour_2, facecolors='none', linewidth=1.2)

ax.set_ylabel('Error')
ax.set_xlabel('Horizon (Years)')
ax.yaxis.grid(True, which='major', linestyle=':', alpha=0.6)
ax.legend(loc='best', frameon=False)
fig.tight_layout()
plt.show()
