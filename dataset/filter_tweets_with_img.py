
import pandas as pd

import os

from tqdm import tqdm



def filter_img(label):

    # All paths
    data_path = "/work/socialmedia/multimodal_dataset/final_dataset/tweet_level/" + label + ".pkl"
    dir_root_path = "/work/socialmedia/multimodal_dataset/MultiModalDataset/"


    df = pd.read_pickle(data_path)

    # print(df.head())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date"])

    print(df.loc[0,"date"], df.loc[df.shape[0]-1,"date"])

    # path_to_img = dir_root_path + 'ppd' + '/' + person_id + '/images/' + img_path

    img_list = []
    for person_id in tqdm(os.listdir(dir_root_path + label + '/')):
        img_list += os.listdir(dir_root_path + label + '/' + person_id + '/')


    df_filtered = df[df["img_path"].isin(img_list)].reset_index(drop=True)

    print(df.shape[0], df_filtered.shape[0])

    return df_filtered

def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''
    count = 0
    for file_name in tqdm(os.listdir(path)):
        try:
            df = pd.concat([df,pd.read_csv(path + file_name,lineterminator='\n', low_memory=False)])
        except:
            df = pd.read_csv(path + file_name,lineterminator='\n', low_memory=False)

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    df.to_pickle(final_file_name,protocol=5)


# labels = ["anxiety"]

# for label in labels:
#     print("\n",label)
#     data_path = "/work/socialmedia/multimodal_dataset/mental_health/tweet_level/csv/" + label + ".csv"
#     final_path = "/work/socialmedia/multimodal_dataset/mental_health/tweet_level/" + label + ".pkl"
#     df = pd.read_csv(data_path, lineterminator='\n', low_memory=False)
#     df.to_pickle("/work/socialmedia/multimodal_dataset/mental_health/tweet_level/" + label + ".pkl", protocol=5)

#     # concat_files(data_path, final_path)


final_filtered_df = pd.DataFrame(columns=["person","date","tweet_id","text","img_path","label"])

for label in ["positive","negative"]:
    print("\n",label)

    df_filtered = filter_img(label)

    final_filtered_df = pd.concat([final_filtered_df,df_filtered])

final_filtered_df = final_filtered_df.drop_duplicates().reset_index(drop=True)
final_filtered_df.to_pickle("/work/socialmedia/multimodal_dataset/final_dataset/tweets_with_img.pkl",protocol=5)
