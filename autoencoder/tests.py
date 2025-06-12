import tensorflow as tf
import autoencoder.scripts as s


def autoencoder_visual_test(model: tf.keras.Model, dataset: tf.data.Dataset, index: int):
    predicted = model.predict(s.flatten(dataset[index]))
    predicted = predicted.reshape(32, 32, 3)
    s.sv(predicted[0], "predicted", "predicted.png")
    s.sv(dataset[index][0], "actual", "actual.png")
    return predicted[0]


def autoencoder_loss_test(model: tf.keras.Model, dataset: tf.data.Dataset):
    loss = model.evaluate(dataset)
    return loss

def autoencoder_accuracy_test(model: tf.keras.Model, dataset: tf.data.Dataset):
