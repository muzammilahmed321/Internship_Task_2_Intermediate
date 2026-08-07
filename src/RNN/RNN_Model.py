from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.optimizers import Adam


class RNNModel:

    def __init__(self):
        self.model = None

    # Build RNN Model
    def build_model(self, input_shape):

        self.model = Sequential()

        # RNN Layer
        self.model.add(
            SimpleRNN(
                units=100,
                activation="tanh",
                input_shape=input_shape
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

        print("\n" + "=" * 50)
        print("RNN MODEL SUMMARY")
        print("=" * 50)

        self.model.summary()

        return self.model