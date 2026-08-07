from LSTM_Training import LSTMTraining
from LSTM_Evaluation import LSTMEvaluation


def main():

    print("KARACHI TEMPERATURE PREDICTION USING LSTM")

    # Train the Model


    trainer = LSTMTraining()

    (
        model,
        history,
        scaler_y,
        x_test,
        y_test

    ) = trainer.run()


    # Evaluate the Model


    evaluator = LSTMEvaluation(

        model=model,

        history=history,

        scaler_y=scaler_y,

        x_test=x_test,

        y_test=y_test

    )

    evaluator.run()


    print("PROJECT COMPLETED SUCCESSFULLY")



if __name__ == "__main__":
    main()