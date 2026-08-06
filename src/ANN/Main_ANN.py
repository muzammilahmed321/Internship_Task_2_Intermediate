from ANN_Training import ANNTraining
from ANN_Evaluation import ANNEvaluation


def main():

    # Training Pipeline
    trainer = ANNTraining()
    model, history, x_test, y_test = trainer.run()

    # Evaluation Pipeline
    evaluator = ANNEvaluation(model, history, x_test, y_test)
    evaluator.run()
    print("\nProject Completed Successfully.")
    print("Model has been saved in models/ANN/")
    print("Graphs have been saved in outputs/")

if __name__ == "__main__":
    main()