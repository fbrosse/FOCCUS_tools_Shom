from sources.abstract_lstm import AbstractLSTM


class LSTM_Vanilla(AbstractLSTM):

    def build_model(self, hp):

        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Input, Dropout
        from keras.constraints import MaxNorm
        from keras.optimizers import Adam

        kc = MaxNorm(hp.Float("kc", min_value=3.0, max_value=3.6, step=0.2))
        rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.2, step=0.1)
        fact_g = round(1 / ((1 - rate)))

        model = Sequential()
        model.add(Input((self.n_input, len(self.estimates))))
        model.add(LSTM(units=fact_g * hp.Choice(f"units_lstm_init", [128, 256, 512]),
                       activation='relu',
                       kernel_constraint=kc,
                       return_sequences=True))

        # Tune the number of layers.
        nb_layers = hp.Int("nb_layers", 2, 3)
        for i in range(nb_layers):
            model.add(Dropout(rate=rate))
            args = {'units': fact_g * hp.Choice(f"units_lstm_{i}", [128, 256, 512]),
                    'activation': 'relu',
                    'kernel_constraint': kc}
            if i != (nb_layers - 1):
                args['return_sequences'] = True
            model.add(LSTM(**args))
        # Tune the number of Dense layers.
        for i in range(hp.Int("num_layers_dense", 1, 2)):
            model.add(
                Dense(
                    units=hp.Choice(f"units_dense_{i}", [128, 256, 512]),
                    activation="relu"
                )
            )
        # Add a final dense layer to format the output
        model.add(Dense(self.n_out))
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss="mean_squared_error",
            metrics=["root_mean_squared_error"],
        )
        return model
