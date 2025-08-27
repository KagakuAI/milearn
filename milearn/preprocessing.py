import numpy as np
from sklearn.preprocessing import MinMaxScaler

class BagScaler:
    pass

class BagMinMaxScaler(BagScaler):
    def __init__(self):
        super().__init__()
        self.scaler = MinMaxScaler()

    def fit(self, x):
        self.scaler.fit(np.vstack(x))

    def transform(self, x):
        x_scaled = x.copy()
        for i, bag in enumerate(x):
            x_scaled[i] = self.scaler.transform(bag)

        return x_scaled

