import os

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.optimizers import Adam

from ANN_Model import ANNModel


class ANNTraining:

    def __init__(self):

        self.model = None
        self.history = None

        self.data = None

        self.x_train = None
        self.y_train = None

        self.x_test = None
        self.y_test = None

        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()


    # Load Dataset

    def load_data(self):
        self.data = pd.read_csv("../../datasets/Churn_Modelling.csv")

        print("\nDataset Loaded Successfully\n")
        print(f"Data Shape : {self.data.shape}")


    # Clean Data

    def clean_data(self):

        df = self.data

        df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
        df.dropna(inplace=True)

        df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
        df['Gender'] = self.encoder.fit_transform(df['Gender'])

        self.data = df

        print("\nData Cleaning Completed")


    # Prepare Train/Test Split + Scaling

    def prepare(self):

        df = self.data

        x = df.drop('Exited', axis=1)
        y = df['Exited']

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42, stratify=y
        )

        self.x_train = self.scaler.fit_transform(self.x_train)
        self.x_test = self.scaler.transform(self.x_test)

        print("\nData Preparation Completed")


    # Build ANN Model

    def build_model(self):

        input_dim = self.x_train.shape[1]

        ann = ANNModel(input_dim)
        self.model = ann.build_model()

        print("\nANN Model Created Successfully")


    # Compile Model

    def compile_model(self):

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel Compiled Successfully")


    # Train Model

    def training_model(self):

        print("\nTraining Started...\n")

        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=32,
            verbose=1
        )

        print("\nTraining Completed")


    # Save Model

    def save_model(self):

        os.makedirs("../../models/ANN", exist_ok=True)
        self.model.save("../../models/ANN/ann_model.keras")

        print("\nModel Saved Successfully")


    # Execute Complete Pipeline

    def run(self):

        self.load_data()
        self.clean_data()
        self.prepare()
        self.build_model()
        self.compile_model()
        self.training_model()
        self.save_model()

        return (
            self.model,
            self.history,
            self.x_test,
            self.y_test
        )