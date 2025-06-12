import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def sv(img, label="test", filename: Path = Path("test.png")):
    """
    Save an image to a file
    """
    plt.imshow(img)
    plt.title("Label: {}".format(label))
    plt.axis('off')
    plt.savefig(filename)
    print("Saved image to", filename)
    plt.close()

def flatten(img):
    """
    Flatten an image
    """
    return np.expand_dims(img, axis=0)