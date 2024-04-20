import pandas as pd

def preprocess():
    # preprocessing
    track_metadata_df = pd.read_csv('music/song_data.csv')
    count_play_df = pd.read_csv('music/10000.txt', sep='\t', header=None, names=['user','song','play_count'])
    unique_track_metadata_df = track_metadata_df.groupby('song_id').max().reset_index()
    user_song_list_count = pd.merge(count_play_df, 
                                    unique_track_metadata_df, how='left', 
                                    left_on='song', 
                                    right_on='song_id')
    user_song_list_count.rename(columns={'play_count':'listen_count'},inplace=True)
    del(user_song_list_count['song_id'])
    return user_song_list_count
