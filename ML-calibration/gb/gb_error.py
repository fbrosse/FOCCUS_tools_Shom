# -*- coding: utf-8 -*-
# General imports
import scipy
import numpy as np
import pandas as pd
import os
import glob
import datetime as dt
import joblib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import statsmodels as sm
import seaborn as sns
from tabulate import tabulate
#
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error as mae, mean_squared_error as mse
from sklearn.inspection import permutation_importance
#
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling  import SMOTE


def read_data(dirs, fmt):
    res_files = glob.glob(os.path.join(dirs, '*{}'.format(fmt)))
    res = {os.path.basename(f): pd.read_csv(f, parse_dates=['time']) 
           for f in res_files}
    return res

np.random.seed(1)
data_dir = '/scratch/work/brossef/data_FOCCUS/csv/out/'
ratios = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0]

for stat, df in st.read_data(data_dir, 'csv').items():

    df = df.dropna()
    stat = os.path.splitext(stat)[0]
    print('Processing station', stat)
    # Learning period: 2021-2023 / Validation: 2024
    end_train = dt.datetime.strptime('2023-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_test = dt.datetime.strptime('2024-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')

    df_train = df.loc[df['time'] < end_train]
    df_test = df.loc[(df['time'] >= end_train) & (df['time'] < end_test)]
    beg = df_train['time'].min().strftime('%Y-%m-%d')
    end = df_train['time'].max().strftime('%Y-%m-%d')
    df_train = df_train.drop(['time'], axis=1).dropna()
    df_test = df_test.drop(['time'], axis=1).dropna()

    y_var = 'surge_error'
    estimates = [x for x in list(df_train) if x != y_var]
    cols_resampled = df_train.columns.to_list()
    q_95 = df_train['surge_obs'].quantile(0.95)
    df_train['>95'] = df_train['surge_obs'] > q_95
    X_test = df_test[estimates]
    y_test = df_test[y_var]

    for ratio in ratios:

        sms = {'RAW': None, 'SMOTE': SMOTE(), 'SMOTEENN': SMOTEENN(), 'SMOTETomek': SMOTETomek()}
        for sm_name, sm in sms.items():

            if sm is not None:
                sm.random_state = 42
                sm.sampling_strategy = ratio

                X = df_train[cols_resampled].values
                y = df_train['>95'].values

                X_res, y_res = sm.fit_resample(X, y)
                data_bal = np.append(X_res, y_res.reshape(-1, 1), 1)
                obs_bal = pd.DataFrame(data=data_bal, columns=[*cols_resampled, '>95'])

            else:
                obs_bal = df_train

            X_train = obs_bal[estimates]
            y_train = obs_bal[y_var]

            if any([x.empty for x in [X_train, y_train, X_test, y_test]]):
                print('Missing data, skipping this station')
                continue

            # Hyper parameters tunning
            param_dist = {'n_estimators': [100, 200, 300],
                        'learning_rate': [0.01, 0.05, 0.1],
                        'max_depth': [3, 4, 5, 7],
                        'subsample': [0.8, 0.9, 1],
                        'min_samples_split': [3, 5, 10, 20],
                        'min_samples_leaf': [10, 15, 20, 50],
                        'max_features': [6, 7, 8, 9, 10],
                        'criterion': ['squared_error', 'friedman_mse']}
            gbr = GradientBoostingRegressor()

            # Fit the random search model
            gsearch = RandomizedSearchCV(gbr,
                                        param_distributions=param_dist,
                                        n_iter=300,  # Adjust the number of iterations based on computational resources
                                        scoring='neg_root_mean_squared_error',
                                        cv=TimeSeriesSplit(n_splits=5),
                                        verbose=1,
                                        n_jobs=-1,  # Use all available processors
                                        random_state=42)

            gsearch.fit(X_train, y_train)
            # print the best parameters
            print('{} - ratio {} - sm {} - best parameters: {}'.format(stat, ratio, sm_name, gsearch.best_params_))

            # Create a variable for the best model
            best_gb = gsearch.best_estimator_
            os.makedirs('./models/', exist_ok=True)  
            ficbestmodel = 'models/GB_{}_{}_{}_{}_{}.joblib'.format(sm_name, beg, end, ratio, stat)
            joblib.dump(best_gb, ficbestmodel)

            # Feature Importance - Learning
            feature_importance = best_gb.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            pos = np.arange(sorted_idx.shape[0]) + 0.5

            result = permutation_importance(best_gb, X_test, y_test, n_repeats=100, random_state=42, n_jobs=2)
            sorted_idx_res = result.importances_mean.argsort()

            os.makedirs('./figs/', exist_ok=True)  
            with PdfPages('figs/Importance_GB_{}_{}_{}_{}_{}.pdf'.format(sm_name, beg, end, ratio, stat)) as pdf:
                plt.figure(figsize=(18, 8))
                plt.subplot(1, 2, 1)
                plt.barh(pos, feature_importance[sorted_idx], align='center')
                plt.yticks(pos, np.array(estimates)[sorted_idx])
                plt.title('Feature Importance (MDI)')

                # Test
                plt.subplot(1, 2, 2)
                plt.boxplot(result.importances[sorted_idx_res].T, vert=False,
                            tick_labels=np.array(estimates)[sorted_idx_res])
                plt.title('Permutation Importance (test set)')
                pdf.savefig()


            # Generate predictions with the best model
            obs_test = df_test['surge_obs']
            mod_test = df_test['surge_mod']
            y_pred = best_gb.predict(X_test) + mod_test

            # Stats pour fortes valeurs de surcotes
            quant = 0.95
            th = np.quantile(df['surge_obs'], quant)
            obs_test_th = obs_test[obs_test > th]
            y_pred_th = y_pred[obs_test > th]
            mod_test_th = mod_test[obs_test > th]

            d = {'period': '{}_{}'.format(beg, end),
                'bias_raw': [np.round(np.mean(obs_test - mod_test), 2)],
                'bias_gb': [np.round(np.mean(obs_test - y_pred), 2)],
                'mae_raw': [np.round(mae(obs_test, mod_test), 2)],
                'mae_gb': [np.round(mae(obs_test, y_pred), 2)],
                'rmse_raw': [np.round(np.sqrt(mse(obs_test, mod_test)), 2)],
                'rmse_gb': [np.round(np.sqrt(mse(obs_test, y_pred)), 2)],
                'bias_raw_q': [np.round(np.mean(obs_test_th - mod_test_th), 2)],
                'bias_gb_q': [np.round(np.mean(obs_test_th - y_pred_th), 2)],
                'mae_raw_q': [np.round(mae(obs_test_th, mod_test_th), 2)],
                'mae_gb_q': [np.round(mae(obs_test_th, y_pred_th), 2)],
                'rmse_raw_q': [np.round(np.sqrt(mse(obs_test_th, mod_test_th)), 2)],
                'rmse_gb_q': [np.round(np.sqrt(mse(obs_test_th, y_pred_th)), 2)]}

            dscore = pd.DataFrame(data=d)
            print('Performance at {} for quantile {} - ratio {} - {}'.format(stat, quant, ratio, sm_name))
            print(tabulate(dscore, headers='keys', tablefmt='pretty'))


            # Series -> DataFrame to plot via seaborn (Scatter et QQ-Plot)
            obs_test = obs_test.to_frame('Surge observations (m)')
            mod_test = mod_test.to_frame('Model surge (m)')
            mod_test['Tolosa-SW'] = 'Original'
            y_pred = y_pred.to_frame('Model surge (m)')
            y_pred['Tolosa-SW'] = 'GB - {}'.format(sm_name)

            data_scatter = pd.concat([pd.concat([obs_test, mod_test], axis=1),
                                    pd.concat([obs_test, y_pred], axis=1)],
                                    ignore_index=True)

            for df_t in [obs_test, mod_test, y_pred]:
                for col in df_t.columns.to_list():
                    _, df_t[col] = scipy.stats.probplot(df_t[col], fit=False)

            data_qq = pd.concat([pd.concat([obs_test, mod_test], axis=1),
                                pd.concat([obs_test, y_pred], axis=1)],
                                ignore_index=True)

            for name, data in {'Scatter': data_scatter, 'QQ_Plot': data_qq}.items():
                with PdfPages('figs/{}_GB_{}_{}_{}_{}_{}.pdf'.format(name, stat, beg, end, ratio, sm_name)) as pdf:
                    sns.set_theme()
                    rp = sns.relplot(data=data, x='Surge observations (m)', y='Model surge (m)', hue='Tolosa-SW',
                                    height=10, legend='auto', linewidth=0)
                    rp.fig.subplots_adjust(top=0.9)
                    rp.fig.suptitle('{} - ratio {}'.format(stat, ratio))
                    plt.axline((0, 0), slope=1, color='black')

                    # Comp min & max
                    tick_min = min(data['Surge observations (m)'].min(), data['Model surge (m)'].min(), 0.0)
                    tick_max = max(data['Surge observations (m)'].max(), data['Model surge (m)'].max())
                    plt.xlim(tick_min, tick_max * 1.1)
                    plt.ylim(tick_min, tick_max * 1.1)

                    pdf.savefig()
                    plt.close()
