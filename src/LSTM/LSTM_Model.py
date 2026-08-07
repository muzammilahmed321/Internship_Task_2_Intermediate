from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


class LSTMModel:

    def __init__(self):

        self.model = None


    # Build LSTM Model


    def build_model(self, input_shape):

        self.model = Sequential()

        # First LSTM Layer
        self.model.add(
            LSTM(
                units=100,
                return_sequences=True,
                input_shape=input_shape
            )
        )

        self.model.add(
            Dropout(0.2)
        )

        # Second LSTM Layer
        self.model.add(
            LSTM(
                units=50,
                return_sequences=False
            )
        )

        self.model.add(
            Dropout(0.2)
        )

        # Dense Hidden Layer
        self.model.add(
            Dense(
                units=25,
                activation="relu"
            )
        )

        # Output Layer
        self.model.add(
            Dense(1)
        )

        # Compile Model
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="mean_squared_error",
            metrics=["mae"]
        )


        print("LSTM MODEL SUMMARY")

        self.model.summary()

        return self.model