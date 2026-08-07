import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


class LSTMEvaluation:

    def __init__(
        self,
        model,
        history,
        scaler_y,
        x_test,
        y_test
    ):

        self.model = model
        self.history = history
        self.scaler_y = scaler_y
        self.x_test = x_test
        self.y_test = y_test

        self.predictions = None


    # Evaluate Model


    def evaluate(self):

        print("\nEvaluating Model...\n")

        loss, mae = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=1
        )

        print(f"\nTest Loss : {loss:.4f}")
        print(f"Test MAE  : {mae:.4f}")


    # Make Predictions


    def predict(self):

        self.predictions = self.model.predict(
            self.x_test
        )

        self.predictions = self.scaler_y.inverse_transform(
            self.predictions
        )

        self.y_test = self.scaler_y.inverse_transform(
            self.y_test
        )

        print("\nPrediction Completed Successfully.")


    # Regression Metrics


    def regression_report(self):

        mae = mean_absolute_error(
            self.y_test,
            self.predictions
        )

        mse = mean_squared_error(
            self.y_test,
            self.predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            self.y_test,
            self.predictions
        )


        print("LSTM MODEL EVALUATION")


        print(f"MAE      : {mae:.2f}")
        print(f"MSE      : {mse:.2f}")
        print(f"RMSE     : {rmse:.2f}")
        print(f"R² Score : {r2:.4f}")


    # Actual vs Predicted Graph

    def plot_predictions(self):

        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(12, 6))

        plt.plot(
            self.y_test,
            label="Actual Temperature"
        )

        plt.plot(
            self.predictions,
            label="Predicted Temperature"
        )

        plt.title("Actual vs Predicted Temperature")

        plt.xlabel("Time")

        plt.ylabel("Temperature (°C)")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/LSTM_Prediction.png"
        )

        plt.show()

        print("\nPrediction Graph Saved Successfully.")


    # Training Loss Graph


    def plot_loss(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history.history["loss"],
            label="Training Loss"
        )

        plt.plot(
            self.history.history["val_loss"],
            label="Validation Loss"
        )

        plt.title("Training vs Validation Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/LSTM_Loss.png"
        )

        plt.show()

        print("\nLoss Graph Saved Successfully.")


    # MAE Graph


    def plot_mae(self):

        plt.figure(figsize=(8, 5))

        plt.plot(
            self.history.history["mae"],
            label="Training MAE"
        )

        plt.plot(
            self.history.history["val_mae"],
            label="Validation MAE"
        )

        plt.title("Training vs Validation MAE")

        plt.xlabel("Epoch")

        plt.ylabel("MAE")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/LSTM_MAE.png"
        )

        plt.show()

        print("\nMAE Graph Saved Successfully.")


    # Run Complete Evaluation

    def run(self):

        self.evaluate()

        self.predict()

        self.regression_report()

        self.plot_predictions()

        self.plot_loss()

        self.plot_mae()

        print("\nEvaluation Completed Successfully.")