from sources.abstract_lstm import AbstractLSTM


class LSTM_Encoder_Decoder(AbstractLSTM):

    def build_model(self, hp):

        from keras.models import Sequential
        from keras.layers import Dense, LSTM, Input, Dropout, RepeatVector, TimeDistributed
        from keras.constraints import MaxNorm
        from keras.optimizers import Adam

        kc = MaxNorm(hp.Float("kc", min_value=3.0, max_value=3.6, step=0.2))

        model = Sequential()
        model.add(Input((self.n_input, len(self.estimates))))
        model.add(LSTM(1024, activation="relu", return_sequences=True, kernel_constraint=kc))

        rate = hp.Float("dropout_rate", min_value=0.0, max_value=0.2, step=0.1)
        fact_g = round(1 / ((1 - rate)))

        # Tune the number of layers.
        nb_layers = hp.Int("nb_layers", 2, 3)
        for i in range(nb_layers):
            model.add(Dropout(rate=rate))
            args = {'units': fact_g * hp.Int(f"units_lstm_{i}", min_value=256, max_value=1024, step=256),
                    'activation': 'relu',
                    'kernel_constraint': kc}
            if i != (nb_layers - 1):
                args['return_sequences'] = True
            model.add(LSTM(**args))
        model.add(RepeatVector(self.n_out))
        # Add the final LSTM layer
        model.add(
            LSTM(
                units=hp.Int("units_out", min_value=128, max_value=512, step=128),
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
        model.add(TimeDistributed((Dense(1))))
        learning_rate = 1e-4
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=["root_mean_squared_error"],
        )
        return model
