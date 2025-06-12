import numpy as np
import config as cfg
import scripts as s


import keras
from keras import layers
from keras.datasets import cifar10

# The Data Set - CIFAR10, 60,000 images, 32x32x3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# The Model - A dense neural network (the naive approach)
model = keras.Sequential(
    name="dense_autoencoder",
    layers=[
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(16*16*3, activation='relu'),
        layers.Dense(32*32*3, activation='sigmoid'),
        layers.Reshape((32, 32, 3))
    ]
)

model.summary()
 
if False:
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss=keras.losses.Poisson(),
        metrics=[keras.metrics.Poisson()]
    )
    history = model.fit(
        x_train,
        x_train,
        batch_size=64,
        epochs=2
    )
    model.save(cfg.MODEL_PATH)
    print("Model saved to", cfg.MODEL_PATH)
    print("Model history:", history.history)
else:
    model.load_weights(cfg.MODEL_PATH)


# Test the model

def test_model(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray):
    predicted = model.predict(s.flatten(x_test[0]))
    predicted = predicted[0].reshape(32, 32, 3)
    s.sv(x_test[0], "actual", cfg.OUTPUT_FOLDER / "actual.png")
    s.sv(predicted[0], "predicted", cfg.OUTPUT_FOLDER / "predicted.png")

test_model(model, x_test, y_test)



