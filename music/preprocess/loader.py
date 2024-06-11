import os
import re

import cv2
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import OmegaConf

from music.model.torch_model import nn_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_config(hierarchy: str):
    with initialize(config_path="../../config", version_base="1.1"):
        cfg = compose(config_name="config")
        return OmegaConf.to_container(cfg[hierarchy], resolve=True)


class RecommendModel:
    """
    Load model discard last Sofmax layer to predict latent vector
    """

    def __init__(self, func):
        self.function = func

    def __call__(self, *args, **kwargs):
        # Initialize model
        model_val = nn_model.to(device)
        # Load model
        model_val.load_state_dict(torch.load("music/model.pt", map_location="cpu"))
        model_val.eval()
        # Discard last Softmax layer
        removed = list(nn_model.children())[:-1]
        new_model = nn.Sequential(*removed)

        images, labels = self.function(*args, **kwargs)
        images = images[:, None, :, :]
        images = images / 255.0
        # Display list of available test songs.
        LIST_SONG = [("0", " ")]
        for k, v in enumerate(np.unique(labels)):
            LIST_SONG.append(("{}".format(k + 1), v))
        return LIST_SONG, images, labels, new_model


@RecommendModel
def load_data():
    filenames = [
        os.path.join("Music_Sliced_Images", f)
        for f in os.listdir("Music_Sliced_Images")
        if f.endswith(".jpg")
    ]
    images = []
    labels = []
    for f in filenames:
        song_variable = re.search(r"Music_Sliced_Images/.*_(.+?).jpg", f).group(1)
        tempImg = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        images.append(cv2.cvtColor(tempImg, cv2.COLOR_BGR2GRAY))
        labels.append(song_variable)

    images = np.array(images)

    return images, labels
