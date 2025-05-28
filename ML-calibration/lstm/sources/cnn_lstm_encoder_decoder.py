from sources.abstract_lstm import AbstractLSTM


class CNN_LSTM_Encoder_Decoder(AbstractLSTM):

    def build_model(self, hp):

        from keras.models import Sequential
        from keras.layers import Dense, Flatten, LSTM, Input, Dropout
        from keras.layers import RepeatVector, TimeDistributed
        from keras.layers import Conv1D, MaxPooling1D
        from keras.constraints import MaxNorm
        from keras.optimizers import Adam

        kc = MaxNorm(hp.Float("kc", min_value=3.0, max_value=3.6, step=0.2))
        filters = hp.Int(f"filters_conv", min_value=256, max_value=1024, step=256)
        ks = 144
        rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.2, step=0.1)
        fact_g = round(1 / ((1 - rate)))

        model = Sequential()
        model.add(Input((self.n_input, len(self.estimates))))
        # Tune the number of layers.
        nb_layer = hp.Int("num_layers", 2, 2)
        for i in range(nb_layer):
            model.add(Dropout(rate=rate))
            model.add(
                Conv1D(
                    filters=fact_g * filters,
                    kernel_size=ks,
                    activation='relu',
                    padding='causal',
                    use_bias=hp.Boolean('use_bias')
                )
            )
      
        model.add(MaxPooling1D(pool_size=ks))
        model.add(Flatten())
        model.add(RepeatVector(self.n_out))
        # Add the final LSTM layer
        model.add(
            LSTM(
                units=hp.Int(f"units_out", min_value=256, max_value=1024, step=256),
                activation="relu",
                kernel_constraint=kc,
                return_sequences=True,
            )
        )
        # Tune the number of Dense layers.
        for i in range(hp.Int("num_layers_dense", 1, 2)):
            model.add(
                TimeDistributed(
                    Dense(
                        units=hp.Int(f"units_dense_{i}", min_value=128, max_value=512, step=128),
                        activation="relu"
                    )
                )
            )
        # Add a final dense layer to format the output
        model.add(TimeDistributed(Dense(self.n_out)))
        model.compile(
            optimizer=Adam(learning_rate=1e-5),
            loss="mean_squared_error",
            metrics=["root_mean_squared_error"],
        )
        return model

    # reshape output into [samples, timesteps, features]
    def get_training_data(self, train):
        train_x, train_y = self.to_supervised(train)
        train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))
        return train_x, train_y

    # reshape output into [samples, timesteps, features]
    def get_testing_data(self, test):
        return self.get_training_data(test)
