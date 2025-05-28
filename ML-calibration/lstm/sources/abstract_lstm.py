from abc import ABC, abstractmethod
from numpy import split, array, concatenate
from pandas import read_csv, DataFrame, concat
import numpy as np


class AbstractLSTM(ABC):
    """ LSTM abstract class """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.div = 144

    def get_data(self, filename, estimates):
        dataset = read_csv(filename, header=0, parse_dates=['time'], index_col=['time'])
        dataset = dataset.resample('10min').mean().dropna()
        return dataset[estimates]

    # split a dataset into train/test sets
    def split_dataset(self, data, train_dt, test_dt):
        ## split
        train = data.loc[(data.index >= train_dt[0]) & (data.index < train_dt[1])]
        test = data.loc[(data.index >= test_dt[0]) & (data.index < test_dt[1])]
        self.val_dates = test.index
        # restructure according to lookback window length
        train = array(split(train, len(train)))
        test = array(split(test, len(test)))
        self.train = train
        self.test = test
        self.x_train, self.y_train = self.get_training_data(train)
        self.x_test,  self.y_test  = self.get_testing_data(test)

    def to_supervised(self, train):
        data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))
        X, y = list(), list()
        in_start = 0
        # step over the entire history one time step at a time
        for _ in range(len(data)):
            # define the end of the input sequence
            in_end = in_start + self.n_input
            out_end = in_end + self.n_out
            # ensure we have enough data for this instance
            if out_end <= len(data):
                X.append(data[in_start:in_end, :])
                y.append(data[in_end:out_end, 0])
            # move along one time step
            in_start += 1
        return array(X), array(y)

    def get_training_data(self, train):
        return self.to_supervised(train)

    def get_testing_data(self, test):
        return self.to_supervised(test)

    @abstractmethod
    def build_model(self, **kwargs):
        pass

