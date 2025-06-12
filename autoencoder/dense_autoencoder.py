from keras import layers, models
from keras.datasets import cifar10
import numpy as np

import config as cfg
import scripts as s

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

input_img = layers.Input(shape=(32, 32, 3))
x = layers.Flatten()(input_img)

# Encoder
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(512, activation='relu')(x)
encoded = layers.Dense(128, activation='relu')(x)

# Decoder
x = layers.Dense(512, activation='relu')(encoded)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(32 * 32 * 3, activation='sigmoid')(x)
decoded = layers.Reshape((32, 32, 3))(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.summary()

# Train
autoencoder.fit(x_train, x_train, epochs=32, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

autoencoder.save(cfg.DENSE_MODEL_PATH)

def test_model(model: models.Model, image: np.ndarray):
    decoded_image = model.predict(image)
    return decoded_image


decoded_image = test_model(autoencoder, x_test[0])

s.sv(img=decoded_image[0], label="Dense-Decoded", filename= cfg.OUTPUT_FOLDER / "dense_decoded.png")
s.sv(img=x_test[0], label="Original", filename= cfg.OUTPUT_FOLDER / "dense_original.png")