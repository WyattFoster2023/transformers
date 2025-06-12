from keras import layers, models
from keras.datasets import cifar10
import numpy as np

import config as cfg
import scripts as s
import train_test as tt

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

autoencoder = models.Model(input_img, decoded, name="Dense-Autoencoder")
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()

tt.train_epoch_series(
    autoencoder, 
    x_train, 
    x_test[0:2], 
    epochs=3
)
