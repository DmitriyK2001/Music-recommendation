import math as mt

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds

from music.preprocess.data_loader import load_config


def compute_svd(urm, K):
    U, s, Vt = svds(urm, K)

    dim = (len(s), len(s))
    S = np.zeros(dim, dtype=np.float32)
    for i in range(0, len(s)):
        S[i, i] = mt.sqrt(s[i])

    U = csc_matrix(U, dtype=np.float32)
    S = csc_matrix(S, dtype=np.float32)
    Vt = csc_matrix(Vt, dtype=np.float32)

    return U, S, Vt


def compute_estimated_matrix(U, S, Vt, uTest, MAX_UID, MAX_PID):
    rightTerm = S * Vt
    max_recommendation = load_config("model")["max_recommendation"]
    estimatedRatings = np.zeros(shape=(MAX_UID, MAX_PID), dtype=np.float16)
    recomendRatings = np.zeros(shape=(MAX_UID, max_recommendation), dtype=np.float16)
    for userTest in uTest:
        prod = U[userTest, :] * rightTerm
        estimatedRatings[userTest, :] = prod.todense()
        recomendRatings[userTest, :] = (-estimatedRatings[userTest, :]).argsort()[
            :max_recommendation
        ]
    return recomendRatings


def show_recomendations(uTest, uTest_recommended_items, small_set, num_recomendations=10):
    rec = []
    for user in uTest:
        rec_user = set()
        print("-" * 70)
        print("Recommendation for user id {}".format(user))
        rank_value = 1
        i = 0
        while rank_value < num_recomendations + 1:
            so = uTest_recommended_items[user, i : i + 1][0]
            if (
                small_set.user[
                    (small_set.so_index_value == so) & (small_set.us_index_value == user)
                ].count()
                == 0
            ):
                song_details = small_set[
                    (small_set.so_index_value == so)
                ].drop_duplicates("so_index_value")[["title", "artist_name"]]
                rec_user.add(list(song_details["title"])[0])
                print(
                    "The number {} recommended song is {} BY {}".format(
                        rank_value,
                        list(song_details["title"])[0],
                        list(song_details["artist_name"])[0],
                    )
                )
                rank_value += 1
            i += 1
        rec.append(rec_user)
    rec_matrix = np.zeros((len(uTest), len(uTest)))
    for i in range(len(rec)):
        for j in range(len(rec)):
            rec_matrix[i][j] = len(rec[i] & rec[j]) / len(rec[i])
    upper_indicies = np.triu_indices(len(uTest), 1)
    upper_elements = rec_matrix[upper_indicies]
    return 1.0 - np.mean(upper_elements)
