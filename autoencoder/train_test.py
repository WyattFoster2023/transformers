import tensorflow as tf
import numpy as np

import config as cfg
import scripts as s


def train_epoch_series(autoencoder: tf.keras.Model, x_train: np.ndarray, x_test: np.ndarray | None = None, epochs: int = 5):
    """
    Train the model for multiple epochs, saving a checkpoint after each.
    If x_test is provided, decoded images will also be saved after each epoch.
    """
    for epoch in range(epochs):
        print(f"Training {autoencoder.name} - Epoch {epoch + 1}/{epochs}")
        autoencoder.fit(
            x_train, x_train,
            epochs=epoch + 1,
            initial_epoch=epoch,
            batch_size=64,
            shuffle=True,
            validation_data=(x_test, x_test) if x_test is not None else None
        )
        
        # Save model checkpoint
        model_path = cfg.MODEL_FOLDER / f"{autoencoder.name}-epoch_{epoch + 1}.keras"
        autoencoder.save(model_path)
        
        # Optionally test and save decoded images
        if x_test is not None:
            for img_index, img in enumerate(x_test):
                decoded_image = autoencoder.predict(s.flatten(img))
                s.sv(
                    img=decoded_image[0],
                    label=f"Image {img_index} - ({autoencoder.name}, epoch {epoch + 1})",
                    filename=cfg.OUTPUT_FOLDER / f"{autoencoder.name}-epoch_{epoch + 1}-index_{img_index}-decoded.png"
                )


def save_original(x_test: np.ndarray):
    for img_index, img in enumerate(x_test):
        s.sv(img=img, label=f"Image {img_index})", filename= cfg.OUTPUT_FOLDER / f"index_{img_index}-original.png")