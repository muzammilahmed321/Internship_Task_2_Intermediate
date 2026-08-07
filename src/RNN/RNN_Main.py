from RNN_Training import RNNTraining
from RNN_Evaluation import RNNEvaluation


def main():


    print("AIRLINE PASSENGER PREDICTION USING RNN")


    # Train the Model

    trainer = RNNTraining()

    model, history, scaler, x_test, y_test = trainer.run()


    # Evaluate the Model


    evaluator = RNNEvaluation(
        model=model,
        history=history,
        scaler=scaler,
        x_test=x_test,
        y_test=y_test
    )

    evaluator.run()

    print("PROJECT COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()