import tensorflow as tf
from tensorflow.keras import layers, Model

class BaseLine(Model):
    def __init__(self, widths = [10,10,5], activation = 'relu'):
        super(BaseLine, self).__init__()
        self.n_layers = len(widths)
        for i in range(self.n_layers):
            setattr(self, f"layer_{i}", layers.Dense(widths[i], activation=activation))
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self,inputs):
        hidden = inputs

        for i in range(self.n_layers):
            layer = getattr(self, f"layer_{i}")
            hidden = layer(hidden)
        
        output = self.output_layer(hidden)
        return output
    
    def model(self):
        x = layers.Input(shape=(14))
        return Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    model = BaseLine()
    model.build((None,14))
    model.model().summary()
