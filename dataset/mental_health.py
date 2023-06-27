import pandas as pd
import json
import os
from tqdm import tqdm

def concat_files(path, final_file_name):
    '''
    Combines all files in a directory and saves it in a single file
    Parameters:
        path (str): directory where all independent files are saved
        final_file_name (str): path of the final file
    '''
    count = 0
    for file_name in os.listdir(path):
        try:
            df = pd.concat([df,pd.read_csv(path + file_name, lineterminator='\n')])
        except:
            df = pd.read_csv(path + file_name, lineterminator='\n')

        count += 1
    df = df.drop_duplicates().reset_index(drop=True)
    print(df.head())
    print(df.shape[0])
    df.to_csv(final_file_name,index=False)



label = "neg"

path = '/work/socialmedia/mentalhealth/data/' + label + '/'

output_path = "/work/socialmedia/multimodal_dataset/mental_health/person_level/" + label + "/"


# ppd_df = pd.read_csv("/work/socialmedia/multimodal_dataset/mental_health/ppd.csv", lineterminator='\n')
# mdd_df = pd.read_csv("/work/socialmedia/multimodal_dataset/mental_health/mdd.csv", lineterminator='\n')
# bipolar_df = pd.read_csv("/work/socialmedia/multimodal_dataset/mental_health/bipolar.csv", lineterminator='\n')
# ocd_df = pd.read_csv("/work/socialmedia/multimodal_dataset/mental_health/ocd.csv", lineterminator='\n')

# df = pd.concat([ppd_df, mdd_df])
# df = pd.concat([df, bipolar_df])
# df = pd.concat([df, ocd_df])


# df = df.drop_duplicates().reset_index(drop=True)

# print(df.head())

# print(ppd_df.shape[0],mdd_df.shape[0],bipolar_df.shape[0],ocd_df.shape[0],df.shape[0])

# df.to_json("/work/socialmedia/multimodal_dataset/mental_health/mental_health.json")

saved_person_data = [x.split(".csv")[0] for x in os.listdir(output_path)]
unsaved_person_data = [x for x in os.listdir(path) if x not in saved_person_data]


for person in tqdm(unsaved_person_data):
    # print(person)
    with open(path + person + "/tweets.json") as f:
        data = json.load(f)

    final_df = {"person":[],"date":[],"tweet_id":[],"text":[],"img_path":[],"label":[]}

    for date in data.keys():
        # print(date)
        for tweet in data[date]:
            tweet_id = tweet["tweet_id"]
            text = tweet["text"]

            # print("\t",tweet_id)

            img_list = [x for x in os.listdir(path + person + "/images/") if x[:len(str(tweet_id))] == str(tweet_id)]
            if(len(img_list) > 0):
                for img in img_list:
                    final_df["person"].append(person)
                    final_df["date"].append(date)
                    final_df["tweet_id"].append(tweet_id)
                    final_df["text"].append(text)
                    final_df["img_path"].append(img)
                    final_df["label"].append(label)
            else:
                final_df["person"].append(person)
                final_df["date"].append(date)
                final_df["tweet_id"].append(tweet_id)
                final_df["text"].append(text)
                final_df["img_path"].append("no_img")
                final_df["label"].append(label)

    person_df = pd.DataFrame(final_df)
    person_df.to_csv(output_path + person + ".csv",index=False)







































