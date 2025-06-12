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
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(32*32*3, activation='sigmoid'),
        layers.Reshape((32, 32, 3))
    ]
)

model.summary()
 
if False:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.MeanSquaredError(),
        metrics=[keras.metrics.MeanSquaredError()]
    )
    history = model.fit(
        x_train,
        x_train,
        batch_size=64,
        epochs=32
    )
    model.save(cfg.MODEL_PATH)
    print("Model saved to", cfg.MODEL_PATH)
    print("Model history:", history.history)
else:
    model = keras.models.load_model(cfg.MODEL_PATH)


# Test the model

def test_model(model: keras.Model, image: np.ndarray):
    print("Actual image shape:", image.shape)

    x = s.flatten(image)
    predicted = model.predict(x)
    predicted = predicted[0]

    print("Predicted image shape:", predicted.shape)

    s.sv(image, "actual", cfg.OUTPUT_FOLDER / "actual.png")
    s.sv(predicted, "predicted", cfg.OUTPUT_FOLDER / "predicted.png")

test_model(model, x_test[1])



