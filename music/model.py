import math as mt

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds


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


def compute_estimated_matrix(urm, U, S, Vt, uTest, K, test, MAX_UID, MAX_PID):
    rightTerm = S * Vt
    max_recommendation = 250
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
    for user in uTest:
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
                print(
                    "The number {} recommended song is {} BY {}".format(
                        rank_value,
                        list(song_details["title"])[0],
                        list(song_details["artist_name"])[0],
                    )
                )
                rank_value += 1
            i += 1
