import os
import re

import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
from hydra import compose, initialize
from omegaconf import OmegaConf

from music.model.torch_model import nn_model


def load_config(hierarchy: str):
    with initialize(config_path="../config", version_base="1.1"):
        cfg = compose(config_name="config")
        return OmegaConf.to_container(cfg[hierarchy], resolve=True)


# config device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri("http://host.docker.internal:5001")


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
        # images = np.expand_dims(images, axis=1)
        images = images[:, None, :, :]
        # print(images.shape)
        images = images / 255.0
        # Display list of available test songs.
        LIST_SONG = [("0", " ")]
        for k, v in enumerate(np.unique(labels)):
            LIST_SONG.append(("{}".format(k + 1), v))
        # print(np.unique(labels))
        # print(LIST_SONG)
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


def recommend_songs(song_name, images, labels, new_model):
    # Initialization latent factors vector size
    matrix_size = load_config("infer")["matrix_size"]
    # Enter a song name which will be an anchor song.)
    recommend_wrt = song_name
    prediction_anchor = torch.zeros((1, matrix_size)).to(device)
    count = 0
    predictions_song = []
    predictions_label = []
    counts = []
    distance_array = []

    with torch.no_grad():
        for i in range(0, len(labels)):
            if labels[i] == recommend_wrt:
                test_image = images[i]
                # test_image = np.expand_dims(test_image, axis=0)
                test_image = test_image[None, :, :, :]
                image_trans = torch.from_numpy(test_image.astype(np.float32)).to(device)
                prediction = new_model(image_trans)
                prediction_anchor = prediction_anchor + prediction
                count = count + 1
            elif labels[i] not in predictions_label:
                predictions_label.append(labels[i])
                test_image = images[i]
                test_image = np.expand_dims(test_image, axis=0)
                image_trans = torch.from_numpy(test_image.astype(np.float32)).to(device)
                prediction = new_model(image_trans)
                predictions_song.append(prediction)
                counts.append(1)
            elif labels[i] in predictions_label:
                index = predictions_label.index(labels[i])
                test_image = images[i]
                test_image = np.expand_dims(test_image, axis=0)
                image_trans = torch.from_numpy(test_image.astype(np.float32)).to(device)
                prediction = new_model(image_trans)
                predictions_song[index] = predictions_song[index] + prediction
                counts[index] = counts[index] + 1
        # Count is used for averaging the latent feature vectors.
        prediction_anchor = prediction_anchor / count
        for i in range(len(predictions_song)):
            predictions_song[i] = predictions_song[i] / counts[i]
            # Cosine Similarity - Computes a similarity score of all songs with respect to the anchor song.
            # cosine = nn.CosineSimilarity(dim=1, eps=1e-6)
            distance_array.append(
                torch.sum(prediction_anchor * predictions_song[i])
                / (
                    torch.sqrt(torch.sum(prediction_anchor**2))
                    * torch.sqrt(torch.sum(predictions_song[i] ** 2))
                )
            )
            # distance_array.append(cosine(prediction_anchor, predictions_song[i]))
        distance_array = torch.tensor(distance_array)
        recommendations = load_config("infer")["recommendations"]
        print("Recommendation is:")
        list_song = []
        choose_song = {
            "id": 1,
            "name": recommend_wrt,
            "link": "templates/music/" + recommend_wrt + ".mp3",
            "genre": "Original Song",
        }
        list_song.append(choose_song)
        # Number of Recommendations is set to 2.
        max_recommendations = load_config("infer")["max_recommendations"]
        while recommendations < max_recommendations:
            index = torch.argmax(distance_array)
            value = distance_array[index]
            print(
                "Song Name: "
                + "'"
                + predictions_label[index]
                + "'"
                + " with value = %f" % (value)
            )
            value = "{:.4f}".format(value.item())
            list_song.append(
                {
                    "id": recommendations,
                    "name": predictions_label[index],
                    "link": "templates/music/" + predictions_label[index] + ".mp3",
                    "genre": "Recommend Song",
                    "metrics": "Similar:",
                    "value": value,
                }
            )
            distance_array[index] = float("-inf")
            recommendations = recommendations + 1
        return list_song


def infer():
    SONGS_CHOICES, images, labels, new_model = load_data()
    for _, k in SONGS_CHOICES[1:5]:
        mlflow.set_experiment(f"infer for {k}")
        song = k
        print(f"Recommendation for {song}")
        recommended = recommend_songs(song, images, labels, new_model)
        for i in range(1, len(recommended)):
            with mlflow.start_run():
                mlflow.log_metric("cosine similarity", recommended[i]["value"])
                mlflow.log_param("recommended_song", recommended[i]["name"])
