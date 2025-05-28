import os
import datetime as dt
import time
import sys
import joblib
import keras_tuner as kt
from keras.callbacks import EarlyStopping
from keras import utils
# Intra-package references
import common
from sources.lstm_vanilla              import LSTM_Vanilla
from sources.lstm_encoder_decoder      import LSTM_Encoder_Decoder
from sources.cnn_lstm_encoder_decoder  import CNN_LSTM_Encoder_Decoder
from sources.conv_lstm_encoder_decoder import Conv_LSTM_Encoder_Decoder

# Set seed value to get reproducible results
seed = 42
utils.set_random_seed(seed)

# Training data
beg_train = dt.datetime.strptime('2021-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_train = dt.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# Validation data
beg_test = dt.datetime.strptime('2023-03-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_test = dt.datetime.strptime('2023-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# Total data for training
beg_tot_train = dt.datetime.strptime('2021-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end_tot_train = dt.datetime.strptime('2023-09-01 00:00:00', '%Y-%m-%d %H:%M:%S')

# Get execution parameters 
params = common.get_params(sys.argv)
# Outputs directory
dir_out = './out/'
project = '{}_{}_{}'.format(params['mod'], params['var'], params['tuner'])

# Get the chosen model 
mod = sys.argv[1]
if mod == 'vanilla':
    mod_cls = LSTM_Vanilla(**params)
if mod == 'enc_dec':
    mod_cls = LSTM_Encoder_Decoder(**params)
if mod == 'cnn':
    mod_cls = CNN_LSTM_Encoder_Decoder(**params)
if mod == 'conv':
    mod_cls = Conv_LSTM_Encoder_Decoder(**params)

# Iterates over the files in the <data> directory (must be *.csv files)
for filename in glob.glob('./data/*.csv'):
    stat = os.path.splitext(os.path.basename(filename))[0]
    # Load the datasets
    data = mod_cls.get_data(filename, params['estimates'])
    # Split into train and test
    mod_cls.split_dataset(data, [beg_train, end_train], [beg_test, end_test])
    tuner = common.get_tuner(params, mod_cls, dir_out, project)
    print('Search space summary\n', tuner.search_space_summary())
    callback = EarlyStopping(min_delta=0.0001, verbose=1, monitor='val_root_mean_squared_error')

    start_time = time.time()
    # Search the hyperparameters 
    tuner.search(
        mod_cls.x_train,
        mod_cls.y_train,
        epochs=params['epochs'],
        validation_data=(mod_cls.x_test, mod_cls.y_test),
        callbacks=[callback]
    )

    # Only retain the n-best 
    models = tuner.get_best_models(num_models=5)
    end_time = time.time()
    print('Build, train & search: {}mn'.format((end_time - start_time) / 60))
    best_model = models[0]
    print('Best model summary\n', best_model.summary())
    print('Tuner results summary\n', tuner.results_summary())

    # Train the best models with the whole dataset 
    # Get the top hyperparameters.
    best_hps = tuner.get_best_hyperparameters(params['num_models'])
    # Build the model with the best hp.
    for i, hps in enumerate(best_hps):
        print('Processing hps', hps.values)
        mod_cls.split_dataset(data, [beg_tot_train, end_tot_train], [beg_test, end_test])
        model = mod_cls.build_model(hps)
        model.fit(
            x=mod_cls.x_train,
            y=mod_cls.y_train,
            epochs=params['epochs'],
            validation_data=(mod_cls.x_test, mod_cls.y_test),
            callbacks=[callback]
        )

        mod_cls.model = model
        # Store the fitted model for post-processing 
        mod_file = '{}/{}/{}_model_{}_{}.keras'.format(dir_out, project, project, i, stat)
        joblib.dump(mod_cls, mod_file)
        print('Model dumped in {}'.format(mod_file))
