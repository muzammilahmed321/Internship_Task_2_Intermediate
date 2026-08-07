import os

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from Data_Preprocessing import DataPreprocessing
from LSTM_Model import LSTMModel



class LSTMTraining:


    def __init__(self):

        self.model = None
        self.history = None


        self.scaler_x = None
        self.scaler_y = None


        self.x_train = None
        self.y_train = None


        self.x_val = None
        self.y_val = None


        self.x_test = None
        self.y_test = None



    # ==================================================
    # LOAD DATA
    # ==================================================

    def load_data(self):


        print("\n" + "=" * 60)
        print("DATA PREPROCESSING STARTED")
        print("=" * 60)


        preprocessing = DataPreprocessing()



        (
            self.x_train,
            self.y_train,

            self.x_val,
            self.y_val,

            self.x_test,
            self.y_test,

            self.scaler_x,
            self.scaler_y

        ) = preprocessing.run()



        print("\nData Loaded Successfully")





    # ==================================================
    # BUILD MODEL
    # ==================================================

    def build_model(self):


        print("\n" + "=" * 60)
        print("BUILDING LSTM MODEL")
        print("=" * 60)



        lstm = LSTMModel()



        self.model = lstm.build_model(

            input_shape=(

                self.x_train.shape[1],

                self.x_train.shape[2]

            )

        )


        self.model.summary()



        print(
            "\nModel Built Successfully"
        )





    # ==================================================
    # TRAIN MODEL
    # ==================================================

    def train_model(self):


        print("\n" + "=" * 60)
        print("MODEL TRAINING STARTED")
        print("=" * 60)



        early_stop = EarlyStopping(

            monitor="val_loss",

            patience=15,

            restore_best_weights=True

        )



        reduce_lr = ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=5,

            min_lr=0.00001

        )



        self.history = self.model.fit(

            self.x_train,

            self.y_train,


            validation_data=(

                self.x_val,

                self.y_val

            ),


            epochs=100,


            batch_size=32,


            callbacks=[

                early_stop,

                reduce_lr

            ],


            verbose=1

        )



        print(
            "\nTraining Completed Successfully"
        )





    # ==================================================
    # SAVE MODEL
    # ==================================================

    def save_model(
            self,
            save_path="outputs"
    ):


        os.makedirs(

            save_path,

            exist_ok=True

        )



        model_path = os.path.join(

            save_path,

            "LSTM_Model.keras"

        )


        self.model.save(
            model_path
        )



        print(
            "\nModel Saved Successfully"
        )


        print(
            model_path
        )





    # ==================================================
    # COMPLETE PIPELINE
    # ==================================================

    def run(self):


        self.load_data()


        self.build_model()


        self.train_model()


        self.save_model()



        return (

            self.model,

            self.history,

            self.scaler_y,

            self.x_test,

            self.y_test

        )