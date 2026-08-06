import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


class ANNEvaluation:

    def __init__(self, model, history, x_test, y_test):
        self.model = model
        self.history = history
        self.x_test = x_test
        self.y_test = y_test
        self.y_pred = None

        os.makedirs("outputs/ANN", exist_ok=True)

    # model evaluation
    def evaluate_model(self):
        loss, acc = self.model.evaluate(self.x_test, self.y_test)
        print(f"\nTest Loss     : {loss:.4f}")
        print(f"Test Accuracy : {acc:.4f}")

        return loss, acc

    # prediction
    def prediction(self):
        y_prob = self.model.predict(self.x_test)
        self.y_pred = (y_prob > 0.5).astype(int)
        return self.y_pred

    # confusion matrix
    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=[10, 5])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig("outputs/ANN/confusion_matrix.png")
        plt.close()

        print("\nConfusion Matrix Saved")

    # Classification Report (Precision, Recall, F1)
    def classfication_report(self):
        report = classification_report(self.y_test, self.y_pred)
        print("classification report")
        print(report)

        f1 = f1_score(self.y_test, self.y_pred, average="macro")
        accuracy = accuracy_score(self.y_test, self.y_pred)
        precision = precision_score(self.y_test, self.y_pred, average="macro")
        recall = recall_score(self.y_test, self.y_pred, average="macro")

        print(f"f1_score={f1:.4f}")
        print(f"acuracy={accuracy:.4f}")
        print(f"recall={recall:.4f}")
        print(f"precision={precision:.4f}")

    # Training History Curves (Accuracy & Loss)
    def plot_training_history(self):

        plt.figure(figsize=[10, 5])
        plt.plot(self.history.history['accuracy'], label='Train Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("outputs/ANN/accuracy_curve.png")
        plt.close()

        plt.figure(figsize=[10, 5])
        plt.plot(self.history.history['loss'], label='Train Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("outputs/ANN/loss_curve.png")
        plt.close()

        print("\nTraining History Plots Saved")

    # Run Complete Evaluation Pipeline
    def run(self):
        self.evaluate_model()
        self.prediction()
        self.plot_confusion_matrix()
        self.classfication_report()
        self.plot_training_history()