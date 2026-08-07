import os

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.callbacks import EarlyStopping

from RNN_Model import RNNModel


class RNNTraining:

    def __init__(self):

        self.model = None
        self.history = None

        self.data = None
        self.dataset = None

        self.scaler = MinMaxScaler(feature_range=(0, 1))

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None


    # Load Dataset

    def load_data(self):

        self.data = pd.read_csv("../../datasets/AirPassengers.csv")


        print("DATASET LOADED SUCCESSFULLY")


        print(f"\nDataset Shape : {self.data.shape}")

        print("\nFirst 5 Records")

        print(self.data.head())

        print("\nLast 5 Records")

        print(self.data.tail())

        print("\nDataset Information")

        self.data.info()

        print("\nMissing Values")

        print(self.data.isnull().sum())

        print("\nStatistical Summary")

        print(self.data.describe())

        # Keep only passenger column
        self.dataset = self.data[['#Passengers']]

        # Normalize data
        scaled_data = self.scaler.fit_transform(self.dataset)

        print("\nData Scaled Successfully.")

        return scaled_data


    # Create Sequences


    def create_dataset(self, dataset, time_step=12):

        X = []
        y = []

        for i in range(len(dataset) - time_step):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])

        return np.array(X), np.array(y)


    # Data Preprocessing


    def preprocess_data(self, scaled_data, time_step=12):

        X, y = self.create_dataset(scaled_data, time_step)

        split_index = int(len(X) * 0.80)

        self.x_train = X[:split_index]
        self.y_train = y[:split_index]

        self.x_test = X[split_index:]
        self.y_test = y[split_index:]

        # Reshape into 3D
        self.x_train = self.x_train.reshape(
            self.x_train.shape[0],
            self.x_train.shape[1],
            1
        )

        self.x_test = self.x_test.reshape(
            self.x_test.shape[0],
            self.x_test.shape[1],
            1
        )

        print("\n" + "=" * 60)
        print("DATA PREPROCESSING COMPLETED")
        print("=" * 60)

        print(f"\nTraining Input Shape : {self.x_train.shape}")
        print(f"Training Output Shape: {self.y_train.shape}")

        print(f"\nTesting Input Shape  : {self.x_test.shape}")
        print(f"Testing Output Shape : {self.y_test.shape}")


    # Build Model

    def build_model(self):

        print("\nBuilding RNN Model...\n")

        rnn = RNNModel()

        self.model = rnn.build_model(
            input_shape=(self.x_train.shape[1], 1)
        )

        print("\nModel Built Successfully.\n")


    # Train Model


    def train_model(self):


        print("MODEL TRAINING STARTED")


        early_stop = EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        )

        self.history = self.model.fit(

            self.x_train,
            self.y_train,

            validation_data=(
                self.x_test,
                self.y_test
            ),

            epochs=100,
            batch_size=8,

            callbacks=[early_stop],

            verbose=1
        )

        print("\nTraining Completed Successfully.")


    # Save Model

    def save_model(self, save_path="outputs"):

        os.makedirs(save_path, exist_ok=True)

        model_path = os.path.join(
            save_path,
            "RNN_Model.keras"
        )

        self.model.save(model_path)

        print(f"\nModel Saved Successfully at:\n{model_path}")


    # Run Complete Pipeline


    def run(self):

        scaled_data = self.load_data()

        self.preprocess_data(scaled_data)

        self.build_model()

        self.train_model()

        self.save_model()

        return (
            self.model,
            self.history,
            self.scaler,
            self.x_test,
            self.y_test
        )