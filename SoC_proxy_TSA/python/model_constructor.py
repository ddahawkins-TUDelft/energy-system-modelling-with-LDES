import calliope
import os

def standardised_model_config(params):

    #auto-config paths and directories
        # path_output_directory = os.path(f"../../results/{params['output_directory_name']}")
        # path_output_netcdf = os.path(f"../../results/{params['output_directory_name']}/{params['output_model_name']}.netcdf")
        path_model_config_yaml = os.path.normpath(f"SoC_proxy_TSA/model_config/{params['config_yaml_name']}.yaml")

    #auto-config calliope logging
        if params['calliope_full_log']: 
            calliope.set_log_verbosity("INFO", include_solver_output=True)

    #auto-config overrides for calliope config

    #initialise dictionary
        calliope_override_dictionary={}
    
    #define horizons
        if 'horizon_start' in params and 'horizon_start' in params:
            calliope_override_dictionary['config.init.time_subset'] = [params['horizon_start'],params['horizon_end']]
    
    #define tvp source
        if 'filename_time_varying_parameters' in params:
            calliope_override_dictionary['data_tables.time_varying_parameters.data'] = f"../data_tables/{params['filename_time_varying_parameters']}.csv"
    
    #provide option for custom override parameters
        if 'dict_additional_overrides' in params:
            calliope_override_dictionary.update(params['dict_additional_overrides'])

    #auto-config calliope model
        model = calliope.Model(
            path_model_config_yaml,
            scenario=params['scenario_name'] if 'scenario_name' in params else 'standard',
            override_dict=calliope_override_dictionary

        )
    
   #export configured calliope model
        return model