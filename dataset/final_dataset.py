
import pandas as pd
import json
import os
from tqdm import tqdm



label = "negative"

path = '/work/socialmedia/multimodal_dataset/MultiModalDataset/' + label + '/'

output_path = "/work/socialmedia/multimodal_dataset/final_dataset/person_level/" + label + "/"

# saved_person_data = [x.split(".csv")[0] for x in os.listdir(output_path)]
# unsaved_person_data = [x for x in os.listdir(path) if x not in saved_person_data]


# for person in tqdm(unsaved_person_data):
#     # print(person)
#     try:
#         with open(path + person + "/timeline.txt") as f:
#             data = f.readlines()

#         final_df = {"person":[],"date":[],"tweet_id":[],"text":[],"img_path":[],"label":[]}
            
#         for record in data:

#             tweet = json.loads(record)

#             tweet_id = tweet["id"]
#             text = tweet["text"]
#             date = tweet["created_at"]

#             img_list = [x for x in os.listdir(path + person + "/") if x[:len(str(tweet_id))] == str(tweet_id)]

#             if(len(img_list) > 0):
#                 for img in img_list:
#                     final_df["person"].append(person)
#                     final_df["date"].append(date)
#                     final_df["tweet_id"].append(tweet_id)
#                     final_df["text"].append(text)
#                     final_df["img_path"].append(img)
#                     final_df["label"].append(label)
#             else:
#                 final_df["person"].append(person)
#                 final_df["date"].append(date)
#                 final_df["tweet_id"].append(tweet_id)
#                 final_df["text"].append(text)
#                 final_df["img_path"].append("no_img")
#                 final_df["label"].append(label)

        
#         person_df = pd.DataFrame(final_df)
#         person_df["date"] = pd.to_datetime(person_df["date"])
#         # print(person_df[person_df["img_path"] != "no_img"])
#         # print(person_df)

#         person_df.to_csv(output_path + person + ".csv",index=False)

#     except:
#         print(f"{person} tweets are not available")


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
            df = pd.concat([df,pd.read_csv(path + file_name, lineterminator='\n')])
        except:
            df = pd.read_csv(path + file_name, lineterminator='\n')

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    print(df.head())
    print(df.shape[0])

    df.to_pickle(final_file_name,protocol=5)




# concat_files(output_path,'/work/socialmedia/multimodal_dataset/final_dataset/tweet_level/' + label + '.pkl')


final_filtered_df = pd.DataFrame(columns=["person","date","tweet_id","text","img_path","label"])
for label in ["positive", "negative"]:

    df = pd.read_pickle('/work/socialmedia/multimodal_dataset/final_dataset/tweet_level/' + label + '.pkl')

    df_filtered = df[df["img_path"] != "no_img"].reset_index(drop=True)

    final_filtered_df = pd.concat([final_filtered_df,df_filtered])

    print(label, df.shape[0], final_filtered_df.shape[0])


final_filtered_df = final_filtered_df.drop_duplicates().reset_index(drop=True)
df.to_pickle("/work/socialmedia/multimodal_dataset/final_dataset/tweets_with_img.pkl",protocol=5)






















































































