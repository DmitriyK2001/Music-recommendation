import os

import mlflow
import pandas as pd
from scipy import io

from music.data_loader import load_config
from music.model import compute_estimated_matrix, compute_svd, show_recomendations
from music.train import train


mlflow.set_tracking_uri("http://host.docker.internal:5001")
mlflow.set_experiment("MRS expetiment")


def infer(K=50, uTest=(1, 2, 3, 4, 5)):
    with mlflow.start_run():
        if not os.path.isfile(load_config("train_infer")["state_matrix_file"]):
            train()
        data_sparse = io.hb_read(load_config("train_infer")["state_matrix_file"])
        small_set = pd.read_csv(load_config("train_infer")["small_set_file"])
        urm = data_sparse
        MAX_PID = urm.shape[1]
        MAX_UID = urm.shape[0]

        U, S, Vt = compute_svd(urm, K)

        uTest_recommended_items = compute_estimated_matrix(
            urm, U, S, Vt, uTest, K, True, MAX_UID, MAX_PID
        )

        perc = show_recomendations(uTest, uTest_recommended_items, small_set)
        mlflow.log_metric("perconalization", perc)
