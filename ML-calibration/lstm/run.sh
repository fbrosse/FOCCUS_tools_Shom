#!/bin/bash

# Set seed value to get reproducible results
export PYTHONHASHSEED=42

# Number of input time steps used to make the prediction  
n_in=144
# Number of output time steps predicted by the model 
n_out=144
# Univariate ('uni') or multivariate ('multi') model type 
types='uni multi'
# Types of LSTM models to train (simple 'vanilla', encoder-decoder LSTM 'enc_dec', CNN-LSTM 'cnn' and 'ConvLSTM 'conv') 
mods='vanilla enc_dec cnn conv'
# Hyperparameters tuner in use in Keras-tuner (HyperBandit 'hyper', RandomizeSearch 'random' and Bayesian Search 'bayes')
tuner='hyper random bayes'
# Max numbers of epochs to build models
epochs=100

# Arguments provided to the 'random' tuner : 1- Max trials, 2- Execution per trail
declare -A random_args=(['vanilla']='200 3' ['enc_dec']='200 3' ['cnn']='200 3' ['conv']='200 3')
# Arguments provided to the 'hyper' tuner : 1- Max number of epochs, 2- Factor btw exploration & exploitation, 3- number of Hyperband pass
declare -A hyper_args=(['vanilla']=$epochs' 5 4' ['enc_dec']=$epochs' 5 4' ['cnn']=$epochs' 5 3' ['conv']=$epochs' 5 3')
# Arguments provided to the 'bayes' tuner : 1- Max trials
declare -A bayes_args=(['vanilla']='200' ['enc_dec']='200' ['cnn']='200' ['conv']='200')

for t in $types; do
    for m in $mods; do
        for tun in $tuner; do 
           args=${tun}_args[$m]
           python kt_lstm.py $m $t $n_in $n_out $tun $epochs ${args[$m]}
	done
    done
done

