def get_params(args):
    params = {
        'mod': args[1],
        'var': args[2],
        'n_input': int(args[3]),
        'n_out': int(args[4]),
        'num_models': 3,
        'tuner': args[5],
        'epochs': int(args[6]),
    }
    if args[5] == 'random':
        params = {
            **params,
            'max_trials': int(args[7]),
            'executions_per_trial': int(args[8]),
        }
    elif args[5] == 'hyper':
        params = {
            **params,
            'max_epochs': int(args[7]),
            'factor': int(args[8]),
            'hyperband_iterations': int(args[9]),
        }
    elif args[5] == 'bayes':
        params = {
            **params,
            'max_trials': int(args[7]),
        }
    # For multivariate purpose ('surge_error' by default)
    if args[2] == 'multi':
        params['estimates'] = ['surge_error', 'surge_obs', 'thetao_s', 'uo_s', 'vo_s', 'zos', 'Hs', 'T0m1']
    else:
        params['estimates'] = ['surge_error']
    return params


def get_tuner(params, mod_cls, dir_out, project):

    import keras_tuner as kt

    if params['tuner'] == 'random':
        tuner = kt.RandomSearch(
            hypermodel=mod_cls.build_model,
            objective="val_root_mean_squared_error",
            max_trials=params['max_trials'],
            executions_per_trial=params['executions_per_trial'],
            overwrite=True,
            directory=dir_out,
            project_name=project,
        )

    elif params['tuner'] == 'hyper':
        tuner = kt.Hyperband(
            hypermodel=mod_cls.build_model,
            objective="val_root_mean_squared_error",
            max_epochs=params['max_epochs'],
            factor=params['factor'],
            hyperband_iterations=params['hyperband_iterations'],
            overwrite=True,
            directory=dir_out,
            project_name=project,
        )

    elif params['tuner'] == 'bayes':
        tuner = kt.BayesianOptimization(
            hypermodel=mod_cls.build_model,
            objective="val_root_mean_squared_error",
            max_trials=params['max_trials'],
            overwrite=True,
            directory=dir_out,
            project_name=project,
        )

    return tuner
