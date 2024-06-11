import pandas as pd
from hydra import compose, initialize
from omegaconf import OmegaConf


def load_config(hierarchy: str):
    with initialize(config_path="../../config", version_base="1.1"):
        cfg = compose(config_name="config")
        return OmegaConf.to_container(cfg[hierarchy], resolve=True)


def preprocess():
    # preprocessing
    track_metadata_df = pd.read_csv(load_config("data_loader")["song_data_file"])
    count_play_df = pd.read_csv(
        load_config("data_loader")["train_file"],
        sep="\t",
        header=None,
        names=["user", "song", "play_count"],
    )
    unique_track_metadata_df = track_metadata_df.groupby("song_id").max().reset_index()
    user_song_list_count = pd.merge(
        count_play_df,
        unique_track_metadata_df,
        how="left",
        left_on="song",
        right_on="song_id",
    )
    user_song_list_count.rename(columns={"play_count": "listen_count"}, inplace=True)
    del user_song_list_count["song_id"]
    return user_song_list_count
