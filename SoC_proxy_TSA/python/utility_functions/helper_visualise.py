import pandas as pd
import calliope 
import numpy as np
import matplotlib
matplotlib.use('TkAgg') #avoids the annoying Qt errors on windows
import matplotlib.pyplot as plt
from cycler import cycler
import mplcursors



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
    y_field,
    path_reference_model: calliope.Model = None,
    ):

    matplotlib.rcParams['axes.prop_cycle'] = cycler(color=plt.cm.plasma(np.linspace(0.0, 0.92, len(list_model_dict))))

    plt.figure(figsize=(12, 6))

    if path_reference_model:
        list_model_dict.append({
            'model': calliope.read_netcdf(path_reference_model),
            'type': 'reference',
            'name': 'reference'
        })
      
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

        elif y_field == 'SoC Proxy':
            y_val = generate_soc_proxy_expost(model_dict['_model']) TODO: this will break on reference model so allow this function to accept arguments directly
        else:
            raise Exception('Invalid variable type for plot')

        if x_field == 'Time':
            # x_list.append(df.index)
            x_val = df.index
        else:
            raise Exception('Invalid variable type for plot')   
        
        if model_type == 'reference':
            line, = plt.plot(x_val, y_val, label=model_name, color = 'grey', zorder=100, picker=5)
        else:
            line, = plt.plot(x_val, y_val, label=model_name, zorder=1, picker=5)

        line._hover_info = {
        "Name": model_name,
        "Type": model_type,
        "Peak": f'{float(np.nanmax(y_val)):.2e}',
        "Peak Date": f'{y_val.idxmax()}',
        }
    
    plt.xlabel(x_field)
    plt.ylabel(y_field)
    plt.title(f'{y_field} vs. {x_field}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    #on hover highlight functions
    ax = plt.gca()
    cursor = mplcursors.cursor(ax.lines, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        info = getattr(sel.artist, "_hover_info", None)
        if info:
            sel.annotation.set_text("\n".join(f"{k}: {v}" for k, v in info.items()))
        else:
            sel.annotation.set_text(sel.artist.get_label())
        sel.annotation.get_bbox_patch().set_alpha(0.9)

    #show
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


def generate_soc_proxy_expost(m):

    from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy

    if m.tsa.type == 'cluster':

        from utility_functions.helper_timeseries_tools import extrapolate_ts_from_cluster_map

        df_timeseries,_ = extrapolate_ts_from_cluster_map(
                    m.paths['cluster_map'], 
                    m.paths['timeseries'])

    
    else:

        from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas

        df_timeseries = calliope_ts_to_pandas(
            source=m.paths['timeseries'],
            date_range_lower_bound=f'{m.calliope_model.params['date_range'][0]}-01-01',
            date_range_upper_bound=f'{m.calliope_model.params['date_range'][-1]}-12-31'
        )

    df_timeseries.set_index('timesteps', inplace=True)
    df_timeseries.columns.name = None 

    df_timeseries,_,_ = generate_soc_proxy(
        df=df_timeseries,
        demand_field=m.tsa.params['name_demand'][0],
        renewables_fields_and_weights= m.soc_proxy.params['capacity_weights'], 
        dispatchable_techs=m.soc_proxy.params['dispatchable_techs'],
        storage_process_losses=m.soc_proxy.params['storage_process_losses'],
        soc_decomposition = m.soc_proxy.params['soc_decomposition'],
        timestamp_col=None
    )
    df_soc_proxy = df_timeseries['soc_proxy_LDES']

    return df_soc_proxy

    

        