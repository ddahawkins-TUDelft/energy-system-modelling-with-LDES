import pandas as pd
import calliope
from typing import Literal
import json
import hashlib
import os
from utility_functions.helper_model_config import clustered_model_config, standardised_model_config
import shutil
from utility_functions.helper_timeseries_tools import calliope_ts_to_pandas
from utility_functions.helper_SoC_proxy_fast_compute import generate_soc_proxy
import numpy as np
from utility_functions.helper_optimisation_tsa import compute_distance_matrix



class tsa_model:
    def __init__(self, path_timeseries: str, description: str = '', tsa_type= Literal["cluster", "optimisation", "none"]):

        # establishing the type
        if tsa_type not in ("cluster", "optimisation","none"):
            raise ValueError(f"Invalid type: {type}")
        

        #create the id and description for this model run
        self.id = ''
        self.description = description
        
        self.calliope_model = calliope_model()
        self.soc_proxy = soc_proxy()
        self.tsa = tsa(type = tsa_type)

        if not os.path.exists(path_timeseries):
            raise Exception(f'The provided timeseries path ({path_timeseries}) does not exist.')

        self.paths = {
            'directory': '',
            'subdirectories': {
                'calliope_models': 'calliope_models',
                'cluster_maps': 'cluster_maps',
                'timeseries': 'timeseries'
                },
            'calliope_model': '',
            'cluster_map': '',
            'timeseries': path_timeseries
        }

    def set_directory(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.paths['directory'] = directory

    def compute_id(self, length=20):
        # Select only the parameters you care about
        relevant = {
            "calliope": self.calliope_model.params,
            "soc_proxy": self.soc_proxy.params,
            "tsa": self.tsa.params
        }
        # Stable serialization
        blob = json.dumps(relevant, sort_keys=True, separators=(",", ":"))
        # Short hash
        self.id = hashlib.sha256(blob.encode()).hexdigest()[:length]
    
    # Function which generates all the path names for various aspects of the model including calliope, cluster_maps, and timeseries
    def assign_paths(self, directory: str = None):
        if directory:
            self.set_directory(directory=directory)

        if not self.paths['directory']:
            raise Exception('No high-level directory has been assigned to the model. Please assign a directory using tsa_model.set_directory() or pass the parameter into this function')
        if not self.id:
            raise Exception('No unique id has been assigned to this model. Please assign an id using tsa_model.compute_id()')
        
        #assign and create the relevant directories
        if not os.path.exists(f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}"):
            os.makedirs(f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}")
        self.paths['calliope_model'] = f"{self.paths['directory']}/{self.paths['subdirectories']['calliope_models']}/{self.id}.netcdf"
        
        if not os.path.exists(f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}"):
            os.makedirs(f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}")
        self.paths['cluster_map'] = f"{self.paths['directory']}/{self.paths['subdirectories']['cluster_maps']}/{self.id}.csv"
        
        #here we copy the source timeseries data into the model's directory and reassign the path id
        if not os.path.exists(f"{self.paths['directory']}/{self.paths['subdirectories']['timeseries']}"):
            os.makedirs(f"{self.paths['directory']}/{self.paths['subdirectories']['timeseries']}")
        shutil.copy2(self.paths['timeseries'], f"{self.paths['directory']}/{self.paths['subdirectories']['timeseries']}/{self.id}.csv")
        self.paths['timeseries'] = f"{self.paths['directory']}/{self.paths['subdirectories']['timeseries']}/{self.id}.csv"
        

    #CALLIOPE FUNCTIONS -------------------------------------------------------------------------------------------------

    def configure_calliope(self):
        #TODO: function that exports or generates the relevant parameters for calliope

        #check id exists
        if not self.id:
            raise Exception('A unique ID has not been assigned, please fully configure the model with appropriate parameters and then call tsa_model.compute_id().')
        print(f'> Configure: Validating and Configuring Calliope model: {self.id}')
        
        #checks model type and checks relevant files exist
        if not os.path.exists(self.paths['timeseries']):
            raise Exception(f'The timeseries file does not exist at path: {self.paths['timeseries']}')
        if self.tsa.type != 'none' and not os.path.exists(self.paths['cluster_map']):
            raise Exception(f'The cluster map file does not exist at path: {self.paths['cluster_map']}')
        if not self.paths['calliope_model']:
            raise Exception('A valid save path has not been configured.')

        #check calliope parameters exist and validate:
        validate_params(
            self.calliope_model.params, 
            allow_extra=True, 
            template= { 
                'type': str,
                'config_yaml_name': str,
                'date_range': list,
                'calliope_full_log': list
            }
        )

        #generate additional parameters necessary for model
        self.calliope_model.params.update({
            'horizon_start': f"{self.calliope_model.params['date_range'][0]}-01-01",
            'horizon_end': f"{self.calliope_model.params['date_range'][-1]}-12-31",
            'output_model_name': self.id,
            'path_netcdf': self.paths['calliope_model'],
            'path_cluster_map': self.paths['cluster_map'],
            'path_timeseries': self.paths['timeseries'],
        })


        self.calliope_model.model = clustered_model_config(self.calliope_model.params)
        self.calliope_model.status = 'configured' 

    def build_calliope(self):
        if not self.paths['calliope_model']:
            raise Exception('A valid save path has not been configured.')
        if os.path.exists(self.paths['calliope_model']):
            print(f'> Build: Results already exist for: {self.id}, loading model instead.')
            self.calliope_model.model = calliope.read_netcdf(self.paths['calliope_model'])
            self.calliope_model.status = 'solved'
        else:
            if self.calliope_model.status != 'configured':
                raise Exception('Calliope model has not been configured.')
            print(f'> Building Calliope model: {self.id}')
            self.calliope_model.model.build()
            self.calliope_model.status = 'built'
        
    def solve_and_save_calliope(self):
        if not self.paths['calliope_model']:
            raise Exception('A valid save path has not been configured.')
        if os.path.exists(self.paths['calliope_model']):
            print('> Solve: skipping...')
            self.calliope_model.model = calliope.read_netcdf(self.paths['calliope_model'])
            self.calliope_model.status = 'solved'
        else:
            if self.calliope_model.status != 'built':
                raise Exception('Calliope model has not been built.')
            print(f'> Solve: Solving Calliope model: {self.id}')
            self.calliope_model.model.solve()
            self.calliope_model.status = 'solved'
            self.calliope_model.model.to_netcdf(self.paths['calliope_model'])
            print(f'> Save: Solution saved to: {self.paths['calliope_model']}')

    #SOC Proxy FUNCTIONS -------------------------------------------------------------------------------------------------
    
    # none necessary

    #TSA FUNCTIONS -------------------------------------------------------------------------------------------------

    def compute_features_dataframe(self):
        print(f'> TSA: Extracting features dataframe for {self.id}') #TODO:

        #load the timeseries
        df_timeseries = calliope_ts_to_pandas(
            self.paths['timeseries'],
            date_range_lower_bound=f"{self.calliope_model.params['date_range'][0]}-01-01",
            date_range_upper_bound=f"{self.calliope_model.params['date_range'][-1]}-12-31"
        )
        df_timeseries.set_index('timesteps', inplace=True)
        df_timeseries.columns.name = None 

        original_columns = df_timeseries.columns.values.tolist()

        #if soc proxy is to be used, append the requested column.
        if self.tsa.params['soc_proxy']['use_soc_proxy']:

            original_columns.extend(self.tsa.params['soc_proxy']['proxy_inputs_to_consider'])

            df_timeseries,_,_ = generate_soc_proxy(
                df=df_timeseries,
                demand_field='demand_power',
                renewables_fields_and_weights= self.soc_proxy.params['capacity_weights'], 
                dispatchable_techs=self.soc_proxy.params['dispatchable_techs'],
                storage_process_losses=self.soc_proxy.params['storage_process_losses'],
                soc_decomposition = self.soc_proxy.params['soc_decomposition'],
                timestamp_col=None
            )
            df_timeseries = df_timeseries[original_columns]

        #if aggregating daily bool is True, then aggregate otherwise transpose the hourly data into daily profiles to reduce MILP load
        if self.tsa.params['resample_to_daily_resolution']:
            # Select only numeric columns (e.g., drop metadata if present)
            df_timeseries = df_timeseries.select_dtypes(include=[np.number])
            # Group by day and apply aggregation
            df_timeseries = df_timeseries.resample("D").agg('mean')
        else:
            # Create containers
            daily_rows = []
            days = []

            # Group by day (normalize keeps midnight timestamps)
            for day, group in df_timeseries.groupby(pd.Grouper(freq="D")):
                if len(group) != 24:
                    # Skip incomplete days (DST or edges)
                    continue
                # Build a 24h vector per variable and concatenate
                row = np.concatenate([group[var].to_numpy() for var in original_columns])
                daily_rows.append(row)
                days.append(day.normalize())

            # Build column names once
            col_names = [f"{var}_h{h:02d}" for var in original_columns for h in range(24)]

            # Build daily dataframe with a proper Date index
            df_timeseries = pd.DataFrame(
                daily_rows,
                columns=col_names,
                index=pd.DatetimeIndex(days, name="timesteps")
            )

        #assign the output
        self.tsa.df_features = df_timeseries


    def compute_distance_matrix(self):
        print(f'> TSA: Creating distance matrix for {self.id}')

        self.tsa.distance_matrix = compute_distance_matrix(
            feature_df= self.tsa.df_features,
            matrix_weights=self.tsa.params['matrix_weights'],
            metric=self.tsa.params['distance_matrix_metric'],
            column_prefixes_renewables=self.tsa.params['names_renewables'],
            column_prefixes_demand=self.tsa.params['name_demand'],
            column_prefixes_proxy=self.tsa.params['soc_proxy']['proxy_inputs_to_consider'] if self.tsa.params['soc_proxy']['use_soc_proxy'] else [],
            proxy_window = self.tsa.params['soc_proxy']['proxy_window'] if self.tsa.params['soc_proxy']['use_soc_proxy'] else None
        )

    def apply_tsa(self):
        print(f'> TSA: Applying TSA for {self.id}')
        #observe whether cluster or opt
        #take in params
        #run the MILP solver
        #save cluster map






class calliope_model:
    def __init__(self, params: dict = None):
        
        self.params = params if params else {}
        self.timeseries = {
            'df': pd.DataFrame,
            'paths': [],
        } 
        self.model = None #calliope model
        self.path = ''
        self.status = ''


    def set_params(self, params: dict):
        self.params = params

    def update_param(self, param: str, value):
        self.params['param'] = value

    def set_timeseries(self, timeseries: pd.DataFrame, path):
        self.timeseries['df'] = timeseries
        if self.timeseries['paths'][-1] != path: #if its already there, ignore
            self.timeseries['paths'].append(path)
    
    def get_latest_timeseries_path(self):
        return self.timeseries['paths'][-1]  

class soc_proxy:
    def __init__(self, params: dict = {}):
        self.params = params

    #function sets the parameters variable
    def set_params(self, params: dict):
        self.params = params

class tsa:
    def __init__(self, type: Literal["cluster", "optimisation", "none"]):

        # establishing the type
        if type not in ("cluster", "optimisation","none"):
            raise ValueError(f"Invalid type: {type}")
        self.type = type
        self.status = False
        
        if type != "none":
            self.params = {} 
            self.cluster_map = pd.DataFrame
            self.df_features = pd.DataFrame
            self.distance_matrix = pd.DataFrame

    #function sets the parameters variable
    def set_params(self, params):
        if self.type == 'none':
            raise Exception('tsa is set to type none, tsa parameters cannot be assigned.')
        else:
            self.params = params
    
    #function that saves the cluster map results and save path
    def set_cluster_map(self, cluster_map: pd.DataFrame):
        if self.type == 'none':
            raise Exception('tsa is set to type none, tsa parameters cannot be assigned.')
        else:
            self.cluster_map = cluster_map
    
    





#general function for verifying an input dictionary contains the right keys and value typès given a template
def validate_params(params: dict, template: dict, allow_extra=False):
    """
    Validate a params dict against a template dict of {key: type or tuple of types}.
    Returns True if valid, raises ValueError otherwise.
    """
    # Check missing keys
    missing = [k for k in template if k not in params]
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    # Check type of each key
    wrong_type = [
        k for k, t in template.items()
        if not isinstance(params[k], t)
    ]
    if wrong_type:
        raise TypeError(
            f"Wrong types for keys: { {k: type(params[k]).__name__ for k in wrong_type} }"
        )

    # Check extra keys
    if not allow_extra:
        extras = [k for k in params if k not in template]
        if extras:
            raise ValueError(f"Unexpected extra keys: {extras}")

    return True