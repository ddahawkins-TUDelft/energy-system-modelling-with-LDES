import calliope
import datetime
import pandas as pd
import plotly.express as px
import yaml
import re
import os
import script_utilities as util #custom module to abstract away some frequently used functions

def run():

    # script configuration
    model_type = "two_year_cluster" # model type
    results_directory = f"weather_year_clustering_LDES_study/results/{model_type}" #save directory for results
    # calliope.set_log_verbosity("INFO", include_solver_output=True) #if True, shows calliope logs as output


    # import weather years
    df_casestudy = (util.read_years_weights(model_type)) #df of years and weights, only years relevant for single year model

    #loop over the case study years, generate and solve models for each, saving the results as netcdf
    for index, values in df_casestudy.iterrows():
        print('Running: ', index, values['years'])
        if index >=1: #for debugging purposes just using the first one

                # dictionary that defines the parameters for determining timeseries and weights
            dict_input = {
                'weather_years': values['years'],
                'weights': values['weights'],
                'timeseries_save_path': 'weather_year_clustering_LDES_study/data_tables/two_year_cluster/time_varying_parameters_temp_two_years.csv',
                'timeseries_weights_save_path': 'weather_year_clustering_LDES_study/data_tables/two_year_cluster/time_varying_parameters_weights_temp_two_years.csv',
            }

            # function that generates timeseries and weights to be used in the model
            df_timestep_weights = util.generate_timestep_weights(dict_input)

            # initialise the model given the above and wider parameters
            model = calliope.Model(
            'weather_year_clustering_LDES_study/model_config/model.yaml',
            scenario='single_year_runs_plan',
            override_dict={
                'config.init.time_subset': [df_timestep_weights['timesteps'].min(), df_timestep_weights['timesteps'].max()],   #change the daterange
                'techs.h2_salt_cavern.number_year_cycles': len(dict_input['weather_years']), #update number of cycles parameters, redundant as this constraint has been removed
                'data_tables.time_varying_parameters.data': "../data_tables/two_year_cluster/time_varying_parameters_temp_two_years.csv", #update location for timeseries data
                #  'techs.h2_salt_cavern.v_surplus_factor': max(values['weights']), #activate this when using model_which_accoutns_for_op_year_surplus.yaml
                #  'techs.battery.v_surplus_factor': max(values['weights']), #activate this when using model_which_accoutns_for_op_year_surplus.yaml
                }
            )

            # modeitfy the model to include weights
            model.inputs.timestep_weights.data=df_timestep_weights['timestep_weights'].to_numpy()

            #build and validate the model
            print(f'Building Model [{','.join(str(x) for x in dict_input['weather_years'])}] with weights: [{':'.join(str(x) for x in dict_input['weights'])}]')
            model.build()
            
            # solve the model
            print('Solving Model...')
            model.solve()

            # identify a save path for the netcdf after solving the model, and save
            save_path = results_directory+f'/unmodified/results_years_[{",".join(str(x) for x in values['years'])}]_weight_[{",".join(str(x) for x in values['weights'])}].netcdf'
            model.to_netcdf(save_path)


            

