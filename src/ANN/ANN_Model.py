from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout

class ANNModel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.model = None

    def build_model(self):
        #create sequential model
        self.model = Sequential(name="Churn_ANN")
        #first hidden layer
        self.model.add(
            Dense(units=6, activation="relu", input_dim=self.input_dim)
        )
        #reduce overfitting
        self.model.add(Dropout(0.2))
        #2nd hidden layer
        self.model.add(
            Dense(units=6, activation="relu")
        )
        #reduce overfitting
        self.model.add(Dropout(0.2))
        #outer layer
        self.model.add(
            Dense(units=1, activation="sigmoid")
        )
        return self.model

    def model_summary(self):
        if self.model is None:
            self.build_model()
        self.model.summary()