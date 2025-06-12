import matplotlib.pyplot as plt
import numpy as np

def sv(img, label="test", filename="test.png"):
    """
    Save an image to a file
    """
    plt.imshow(img)
    plt.title("Label: {}".format(label))
    plt.axis('off')
    plt.savefig(filename)
    plt.close()

def flatten(img):
    """
    Flatten an image
    """
    return np.expand_dims(img, axis=0)