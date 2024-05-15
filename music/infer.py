import os

import pandas as pd
from scipy import io

from music.model import compute_estimated_matrix, compute_svd, show_recomendations
from music.train import train


def infer(K=50, uTest=(1, 2, 3, 4, 5)):
    if not os.path.isfile("music/state_matrix.csv"):
        train()
    data_sparse = io.hb_read("music/state_matrix.csv")
    small_set = pd.read_csv("music/small_set.csv")
    urm = data_sparse
    MAX_PID = urm.shape[1]
    MAX_UID = urm.shape[0]

    U, S, Vt = compute_svd(urm, K)

    uTest_recommended_items = compute_estimated_matrix(
        urm, U, S, Vt, uTest, K, True, MAX_UID, MAX_PID
    )

    show_recomendations(uTest, uTest_recommended_items, small_set)

    uTest = [0]
    # Get estimated rating for test user
    print("Predictied ratings:")
    uTest_recommended_items = compute_estimated_matrix(
        urm, U, S, Vt, uTest, K, True, MAX_UID, MAX_PID
    )
    show_recomendations(uTest, uTest_recommended_items, small_set)
