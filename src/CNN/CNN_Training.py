import os

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam

from CNN_Model import CNNModel


class CNNTraining:

    def __init__(self):

        self.model = None
        self.history = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None


    # Load CIFAR-10 Dataset

    def load_dataset(self):

        (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

        print("\nDataset Loaded Successfully\n")

        print(f"Training Images : {self.x_train.shape}")
        print(f"Training Labels : {self.y_train.shape}")
        print(f"Testing Images  : {self.x_test.shape}")
        print(f"Testing Labels  : {self.y_test.shape}")


    # Normalize Images

    def preprocess_data(self):

        self.x_train = self.x_train.astype("float32") / 255.0
        self.x_test = self.x_test.astype("float32") / 255.0

        print("\nData Preprocessing Completed")


    # Create CNN Model

    def build_model(self):

        cnn = CNNModel()

        self.model = cnn.build_model()

        print("\nCNN Model Created Successfully")


    # Compile Model

    def compile_model(self):

        self.model.compile(
            optimizer=Adam(),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        print("\nModel Compiled Successfully")


    # Train Model
  
    def train_model(self):

        print("\nTraining Started...\n")

        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_data=(self.x_test, self.y_test),
            epochs=10,
            batch_size=32,
            verbose=1
        )

        print("\nTraining Completed")


    # Save Model

    def save_model(self):

        os.makedirs("models/CNN", exist_ok=True)

        self.model.save("models/CNN/cnn_model.keras")

        print("\nModel Saved Successfully")


    # Execute Complete Pipeline

    def run(self):

        self.load_dataset()

        self.preprocess_data()

        self.build_model()

        self.compile_model()

        self.train_model()

        self.save_model()

        return (
            self.model,
            self.history,
            self.x_test,
            self.y_test
        )