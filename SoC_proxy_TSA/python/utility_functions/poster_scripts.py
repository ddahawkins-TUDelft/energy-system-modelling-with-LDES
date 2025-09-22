import pandas as pd
import helper_timeseries_tools as tt
import matplotlib.pyplot as plt
import numpy as np


df = tt.calliope_ts_to_pandas(
            'SoC_proxy_TSA/data/timeseries/time_varying_parameters.csv',
            date_range_lower_bound='2015-01-01',
            date_range_upper_bound='2020-12-31',
        )

df.set_index('timesteps', inplace=True)

df_daily = df.resample("D").agg('mean')
df_daily['combined']=np.mean(df['offshore_wind'],df['onshore_wind'],df['solar'])

plt.figure(figsize=(12, 6))
plt.scatter(x=df.index, y= df['combined'])
plt.xlabel('Time')
plt.ylabel('SoC')
plt.title('Wind')
plt.grid(True)
# plt.legend()
plt.tight_layout()
plt.show()

print('done')