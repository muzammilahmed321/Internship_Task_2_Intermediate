import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)


class CNNEvaluation:

    def __init__(self, model, history, x_test, y_test):

        self.model = model
        self.history = history
        self.x_test = x_test
        self.y_test = y_test

        self.class_names = [
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck"
        ]
   # Evaluate Model

    def evaluate(self):

        print("\nEvaluating Model...\n")

        loss, accuracy = self.model.evaluate(
            self.x_test,
            self.y_test,
            verbose=1
        )

        print(f"\nTest Loss     : {loss:.4f}")
        print(f"Test Accuracy : {accuracy:.2%}")


    # Predict Classes


    def predict_classes(self):

        predictions = self.model.predict(self.x_test)

        predicted_labels = np.argmax(
            predictions,
            axis=1
        )

        return predicted_labels


    # Classification Report


    def classification_report(self):

        predicted_labels = self.predict_classes()

        print("\nClassification Report\n")

        print(
            classification_report(
                self.y_test,
                predicted_labels,
                target_names=self.class_names
            )
        )


    # Confusion Matrix


    def confusion_matrix(self):

        predicted_labels = self.predict_classes()

        cm = confusion_matrix(
            self.y_test,
            predicted_labels
        )

        os.makedirs("outputs", exist_ok=True)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )

        fig, ax = plt.subplots(figsize=(10, 10))

        display.plot(
            ax=ax,
            cmap="Blues",
            colorbar=False
        )

        plt.title("Confusion Matrix")

        plt.savefig(
            "outputs/confusion_matrix.png"
        )

        plt.show()


    # Accuracy Graph

    def plot_accuracy(self):

        os.makedirs("outputs", exist_ok=True)

        plt.figure(figsize=(8,5))

        plt.plot(
            self.history.history["accuracy"],
            label="Training Accuracy"
        )

        plt.plot(
            self.history.history["val_accuracy"],
            label="Validation Accuracy"
        )

        plt.title("Model Accuracy")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/accuracy.png"
        )

        plt.show()


    # Loss Graph


    def plot_loss(self):

        plt.figure(figsize=(8,5))

        plt.plot(
            self.history.history["loss"],
            label="Training Loss"
        )

        plt.plot(
            self.history.history["val_loss"],
            label="Validation Loss"
        )

        plt.title("Model Loss")

        plt.xlabel("Epoch")

        plt.ylabel("Loss")

        plt.legend()

        plt.grid(True)

        plt.savefig(
            "outputs/loss.png"
        )

        plt.show()


    # Run Evaluation


    def run(self):

        self.evaluate()

        self.classification_report()

        self.confusion_matrix()

        self.plot_accuracy()

        self.plot_loss()

        print("\nEvaluation Completed Successfully.")