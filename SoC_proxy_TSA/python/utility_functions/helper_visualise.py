import pandas as pd
import calliope 
import numpy as np
from scipy.spatial.distance import cdist
import pyomo.environ as pyo
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import utility_functions.helper_timeseries_tools as tt
import matplotlib
matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
import matplotlib.pyplot as plt

def visualise_soc(model, cluster_params: dict = None):

    df = pd.DataFrame

    if cluster_params:
        # path_timeseries = cluster_params['path_timeseries'] #for later when i add soc proxy as an option to this function TODO:

        df_clustermap = pd.read_csv(cluster_params['path_cluster_map'])

        df_clustermap = df_clustermap.rename(columns={
        'timesteps': 'datesteps',
        'PeriodNum': 'mapped_datesteps'
        })
        df_clustermap['datesteps'] = pd.to_datetime(df_clustermap['datesteps'], format='%Y-%m-%d')
        df_clustermap['mapped_datesteps'] = pd.to_datetime(df_clustermap['mapped_datesteps'], format='%Y-%m-%d')

        #pull the intracluster soc
        df_intracluster_soc = (   
                (model.results['storage'].fillna(0))
                .to_series()
                # .where(lambda x: x != 0)
                .dropna()
                .to_frame('intra_soc')
                .reset_index()
            )
        df_intracluster_soc=df_intracluster_soc[df_intracluster_soc['techs'] == 'h2_salt_cavern']
        df_intracluster_soc['mapped_datesteps'] = pd.to_datetime(df_intracluster_soc['timesteps'], format='%Y-%m-%d')


        #pull the intercluster soc
        df_intercluster_soc = (   
            (model.results['storage_inter_cluster'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('inter_soc')
            .reset_index()
        )
        df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
        df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
        
        #merge everything and filter
        df = df_intercluster_soc.merge(df_clustermap, on='datesteps', how='left')
        df  = df .merge(df_intracluster_soc, on='mapped_datesteps', how='left')
        df  = df [['datesteps','timesteps','inter_soc','intra_soc']]

        #create a proper measure of timestamps
        time_only = df ['timesteps'].dt.time
        df ['full_timestamp'] = df ['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))
        
        
        #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
        df ['soc'] = df ['inter_soc']+df ['intra_soc']
        df .drop(['datesteps','timesteps','inter_soc','intra_soc'], axis=1, inplace=True)
        df .rename(columns={'full_timestamp': 'timesteps' }, inplace=True)
        df  = df .set_index('timesteps')
        
    else:
        #soc if full model
        df = (   
                (model.results['storage'].fillna(0))
                .to_series()
                # .where(lambda x: x != 0)
                .dropna()
                .to_frame('soc')
                .reset_index()
            )
        df=df[df['techs'] == 'h2_salt_cavern']
        df = df.set_index('timesteps')
        df.drop(['nodes','techs'], axis=1, inplace=True)

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['soc'], label='State of Charge')
    plt.xlabel('Time')
    plt.ylabel('SoC')
    plt.title('Storage State of Charge Over Time')
    plt.grid(True)
    # plt.legend()
    plt.tight_layout()
    plt.show()

    return df
    
def visualise(
    list_model_dict: list,
    x_field,
    y_field
    ):

    plt.figure(figsize=(12, 6))
      
    for model_dict in list_model_dict:
        model = model_dict['model']
        model_type = model_dict['type']
        model_name = model_dict['name']

        if model_type == 'clustered':
            cluster_params = model_dict['cluster_params']
        else:
            cluster_params = None

        df = get_df(model, cluster_params=cluster_params)
        
        if y_field == 'State of Charge':
            # y_list.append(df['soc'])
            y_val = df['soc']
        else:
            raise Exception('Invalid variable type for plot')

        if x_field == 'Time':
            # x_list.append(df.index)
            x_val = df.index
        else:
            raise Exception('Invalid variable type for plot')   
        if model_type == 'reference':
            plt.plot(x_val, y_val, label=model_name, colour = 'black', zorder=100)
        else:
            plt.plot(x_val, y_val, label=model_name, zorder=1)

    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title(f'{y_field} vs. {x_field}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()         

def get_df(model, cluster_params: dict = None):

    df = pd.DataFrame

    if cluster_params:
        # path_timeseries = cluster_params['path_timeseries'] #for later when i add soc proxy as an option to this function TODO:

        df_clustermap = pd.read_csv(cluster_params['path_cluster_map'])

        df_clustermap = df_clustermap.rename(columns={
        'timesteps': 'datesteps',
        'PeriodNum': 'mapped_datesteps'
        })
        df_clustermap['datesteps'] = pd.to_datetime(df_clustermap['datesteps'], format='%Y-%m-%d')
        df_clustermap['mapped_datesteps'] = pd.to_datetime(df_clustermap['mapped_datesteps'], format='%Y-%m-%d')

        #pull the intracluster soc
        df_intracluster_soc = (   
                (model.results['storage'].fillna(0))
                .to_series()
                # .where(lambda x: x != 0)
                .dropna()
                .to_frame('intra_soc')
                .reset_index()
            )
        df_intracluster_soc=df_intracluster_soc[df_intracluster_soc['techs'] == 'h2_salt_cavern']
        df_intracluster_soc['mapped_datesteps'] = pd.to_datetime(df_intracluster_soc['timesteps'], format='%Y-%m-%d')


        #pull the intercluster soc
        df_intercluster_soc = (   
            (model.results['storage_inter_cluster'].fillna(0))
            .to_series()
            # .where(lambda x: x != 0)
            .dropna()
            .to_frame('inter_soc')
            .reset_index()
        )
        df_intercluster_soc=df_intercluster_soc[df_intercluster_soc['techs'] == 'h2_salt_cavern']
        df_intracluster_soc['mapped_datesteps'] = df_intracluster_soc['mapped_datesteps'].dt.normalize()
        
        #merge everything and filter
        df = df_intercluster_soc.merge(df_clustermap, on='datesteps', how='left')
        df  = df .merge(df_intracluster_soc, on='mapped_datesteps', how='left')
        df  = df [['datesteps','timesteps','inter_soc','intra_soc']]

        #create a proper measure of timestamps
        time_only = df ['timesteps'].dt.time
        df ['full_timestamp'] = df ['datesteps'].dt.normalize() + pd.to_timedelta(time_only.astype(str))
        
        
        #compute a comprehensive SoC, combining intracluster variatinos and intercluster variations
        df ['soc'] = df ['inter_soc']+df ['intra_soc']
        df .drop(['datesteps','timesteps','inter_soc','intra_soc'], axis=1, inplace=True)
        df .rename(columns={'full_timestamp': 'timesteps' }, inplace=True)
        df  = df .set_index('timesteps')
        
    else:
        #soc if full model
        df = (   
                (model.results['storage'].fillna(0))
                .to_series()
                # .where(lambda x: x != 0)
                .dropna()
                .to_frame('soc')
                .reset_index()
            )
        df=df[df['techs'] == 'h2_salt_cavern']
        df = df.set_index('timesteps')
        df.drop(['nodes','techs'], axis=1, inplace=True)
    
    return df




        