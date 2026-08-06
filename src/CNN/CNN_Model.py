from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


class CNNModel:

    def __init__(self):
        self.model = None

    def build_model(self):

        # Create Sequential CNN Model
        self.model = Sequential(name="CIFAR10_CNN")


        # First Convolution Block

        self.model.add(
            Conv2D(
                filters=32,
                kernel_size=(3, 3),
                padding="same",
                activation="relu",
                input_shape=(32, 32, 3)
            )
        )

        self.model.add(
            MaxPooling2D(pool_size=(2, 2))
        )


        # Second Convolution Block

        self.model.add(
            Conv2D(
                filters=64,
                kernel_size=(3, 3),
                padding="same",
                activation="relu"
            )
        )

        self.model.add(
            MaxPooling2D(pool_size=(2, 2))
        )


        # Convert Feature Maps to Vector

        self.model.add(
            Flatten()
        )


        # Fully Connected Layer

        self.model.add(
            Dense(
                units=128,
                activation="relu"
            )
        )

        # Reduce Overfitting
        self.model.add(
            Dropout(0.5)
        )

        # Output Layer
        self.model.add(
            Dense(
                units=10,
                activation="softmax"
            )
        )

        return self.model

    def summary(self):

        if self.model is None:
            self.build_model()

        self.model.summary()