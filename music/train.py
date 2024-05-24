import mlflow
import pandas as pd
from scipy import io
from scipy.sparse import coo_matrix

from music.data_loader import load_config, preprocess


mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("MRS expetiment")


def train():
    with mlflow.start_run():
        user_song_list_count = preprocess()
        user_song_list_listen = (
            user_song_list_count[["user", "listen_count"]]
            .groupby("user")
            .sum()
            .reset_index()
        )
        user_song_list_listen.rename(
            columns={"listen_count": "total_listen_count"}, inplace=True
        )
        user_song_list_count_merged = pd.merge(
            user_song_list_count, user_song_list_listen
        )
        user_song_list_count_merged["fractional_play_count"] = (
            user_song_list_count_merged["listen_count"]
            / user_song_list_count_merged["total_listen_count"]
        )

        user_codes = user_song_list_count_merged.user.drop_duplicates().reset_index()
        user_codes.rename(columns={"index": "user_index"}, inplace=True)
        user_codes["us_index_value"] = list(user_codes.index)

        song_codes = user_song_list_count_merged.song.drop_duplicates().reset_index()
        song_codes.rename(columns={"index": "song_index"}, inplace=True)
        song_codes["so_index_value"] = list(song_codes.index)

        small_set = pd.merge(user_song_list_count_merged, song_codes, how="left")
        small_set = pd.merge(small_set, user_codes, how="left")
        mat_candidate = small_set[
            ["us_index_value", "so_index_value", "fractional_play_count"]
        ]

        data_array = mat_candidate.fractional_play_count.values
        row_array = mat_candidate.us_index_value.values
        col_array = mat_candidate.so_index_value.values

        data_sparse = coo_matrix((data_array, (row_array, col_array)), dtype=float)
        io.hb_write(load_config("train_infer")["state_matrix_file"], data_sparse)
        small_set.to_csv(load_config("train_infer")["small_set_file"], index=False)
        mlflow.log_param("my", "param")
        return data_sparse
