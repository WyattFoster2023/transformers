from keras import layers, models
from keras.datasets import cifar10
import numpy as np

import config as cfg
import scripts as s

(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

input_img = layers.Input(shape=(32, 32, 3))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

autoencoder.summary()


# Train
if False:
    autoencoder.fit(x_train, x_train, epochs=1, batch_size=64, shuffle=True, validation_data=(x_test, x_test))
else:
    autoencoder = models.load_model(cfg.CONVOLUTIONAL_MODEL_PATH)

test_image = x_test[0]

print(test_image.shape)

decoded_image = autoencoder.predict(s.flatten(test_image))
s.sv(img=decoded_image[0], label="Conv-Decoded", filename= cfg.OUTPUT_FOLDER / "conv_decoded.png")
s.sv(img=test_image, label="Original", filename= cfg.OUTPUT_FOLDER / "conv_original.png")