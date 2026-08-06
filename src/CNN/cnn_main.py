from CNN_Training import CNNTraining
from CNN_Evaluation import CNNEvaluation


def main():


    print("CNN IMAGE CLASSIFICATION PROJECT")



    # Train CNN Model

    trainer = CNNTraining()

    model, history, x_test, y_test = trainer.run()


    # Evaluate Model

    evaluator = CNNEvaluation(
        model=model,
        history=history,
        x_test=x_test,
        y_test=y_test
    )

    evaluator.run()

    print("\nProject Completed Successfully.")
    print("Model has been saved in models/CNN/")
    print("Graphs have been saved in outputs/")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print("\nProgram stopped by user.")

    except Exception as error:

        print("\nAn error occurred:")
        print(error)