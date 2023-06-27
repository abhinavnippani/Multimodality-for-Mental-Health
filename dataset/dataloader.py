import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from PIL import Image
import re



def load_inference_data(data_path, dir_root_path, img_size, batch_size=128):
    '''
    Loads and preprocesses data from a CSV file, creates a custom dataset, and returns a dataloader.

    Args:
        data_path (str): Path to the CSV file containing the data.
        dir_root_path (str): Root directory path where the images are located.
        img_size (int): Desired size of the images.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 128.

    Returns:
        mental_health_dataloader (DataLoader): Dataloader for the preprocessed dataset.
        num_classes (int): Number of unique classes in the dataset.
    '''

    dtype = {"person": str, "tweet_id": str, "text": str} # warning solver

    # Load the data
    data_df = pd.read_pickle(data_path)

    # Don't consider records not having an image
    new_data_df = data_df[data_df.img_path != 'no_img'].reset_index(drop=True)
    print(f"Image count changed from {data_df.shape[0]} to {new_data_df.shape[0]}")

    print(np.unique(new_data_df['label']))
    sum_df = new_data_df.groupby(["label"],as_index=False)["text"].count()
    print(sum_df)

    # Split the data into train and test, stratified by the target variable
    train_data, test_data = train_test_split(new_data_df, test_size=0.2, random_state=200, stratify=new_data_df['label'])

    # Split the train data into train and validation, stratified by the target variable
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=200, stratify=test_data['label'])

    # Reset the indices
    train_data.reset_index(drop=True, inplace=True)
    val_data.reset_index(drop=True, inplace=True)
    test_data.reset_index(drop=True, inplace=True)

    print("\nTrain-Val-Test Split Completed!!")
    print(f"Train Data: {train_data.shape[0]} records, {np.unique(train_data['label'])}")
    print(f"Val Data: {val_data.shape[0]} records, {np.unique(val_data['label'])}")
    print(f"Test Data: {test_data.shape[0]} records, {np.unique(test_data['label'])}")
    
    # Create a dataset
    train_dataset = InferenceDataset(train_data, dir_root_path, img_size)
    val_dataset = InferenceDataset(val_data, dir_root_path, img_size)
    test_dataset = InferenceDataset(test_data, dir_root_path, img_size)

    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Number of unique classes in the data
    num_classes = new_data_df['label'].nunique()

    return train_dataloader, val_dataloader, test_dataloader, num_classes


def load_user_inference_data(data_path, dir_root_path, img_size, batch_size=128):
    '''
    Loads and preprocesses data from a CSV file, creates a custom dataset, and returns a dataloader.

    Args:
        data_path (str): Path to the CSV file containing the data.
        dir_root_path (str): Root directory path where the images are located.
        img_size (int): Desired size of the images.
        batch_size (int, optional): Batch size for the dataloader. Defaults to 128.

    Returns:
        mental_health_dataloader (DataLoader): Dataloader for the preprocessed dataset.
        num_classes (int): Number of unique classes in the dataset.
    '''

    # Load the data
    data_df = pd.read_pickle(data_path)

    # Don't consider records not having an image
    new_data_df = data_df[data_df.img_path != 'no_img'].reset_index(drop=True)
    print(f"Image count changed from {data_df.shape[0]} to {new_data_df.shape[0]}")

    user_condition_mapping = new_data_df[["person","label"]].drop_duplicates()

    # Split the data into train and test, stratified by the target variable
    train_mapping, test_mapping = train_test_split(user_condition_mapping, test_size=0.2, random_state=200, stratify=user_condition_mapping['label'])

    # Split the train data into train and validation, stratified by the target variable
    val_mapping, test_mapping = train_test_split(test_mapping, test_size=0.5, random_state=200, stratify=test_mapping['label'])

    # Reset the indices
    train_mapping.reset_index(drop=True, inplace=True)
    val_mapping.reset_index(drop=True, inplace=True)
    test_mapping.reset_index(drop=True, inplace=True)

    print("\nTrain-Val-Test Split Completed!!")
    print(f"Train Data: {train_mapping.shape[0]} records, {np.unique(train_mapping['label'])}")
    print(f"Val Data: {val_mapping.shape[0]} records, {np.unique(val_mapping['label'])}")
    print(f"Test Data: {test_mapping.shape[0]} records, {np.unique(test_mapping['label'])}")

    # Create a dataset
    train_dataset = InferenceDatasetUser(train_mapping, new_data_df, dir_root_path, img_size)
    val_dataset = InferenceDatasetUser(val_mapping, new_data_df, dir_root_path, img_size)
    test_dataset = InferenceDatasetUser(test_mapping, new_data_df, dir_root_path, img_size)

    # Create a dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Number of unique classes in the data
    num_classes = new_data_df['label'].nunique()

    return train_dataloader, val_dataloader,test_dataloader, num_classes


#----------------------- Inference Data -----------------------------#

class InferenceDataset(Dataset):
    '''
    Custom dataset class for loading and preprocessing images from the Inference Dataset

    Args:
        data (DataFrame): DataFrame containing information about the dataset.
        dir_root_path (str): Root directory path where the images are located.
        img_size (int): Desired size of the images.

    Methods:
        __len__(): Returns the length of the dataset.
        __getitem__(idx): Loads and preprocesses an individual sample from the dataset. 
                          Creates a random image if there is no image present.
    '''

    def __init__(self, data, dir_root_path, img_size=224):
        self.data = data
        self.dir_root_path = dir_root_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # Extract information from dataframe
        img_path = self.data.loc[idx,'img_path']
        person_id = self.data.loc[idx,'person']
        label_name = self.data.loc[idx,'label']
        text = self.data.loc[idx,'text']

        # ============ Image processing ============
        # Account for exceptions where there are no images (FILENOTFOUND) or corrupted at the path.
        try:
            path_to_img = self.dir_root_path + str(label_name) + '/' + str(person_id) + '/' + str(img_path) # get the full path to image
            image = Image.open(path_to_img).convert('RGB')
        except:
            # print("Random IMG Generated")
            image = create_random_image(self.img_size)

        # Apply transforms
        image = self.transform(image)
        text = clean_text(text)

        output = self.data['label'].unique().tolist().index(label_name)

        data = {'img':image, 'text':text, 'output':output}

        return data


#------------------------ Inference Data User ---------------------------------#

class InferenceDatasetUser(Dataset):
    """
    Custom PyTorch dataset class for inference on user-level data.

    Args:
        mapping (DataFrame): Mapping dataframe containing person and label information.
        data (DataFrame): Data dataframe containing tweet data.
        dir_root_path (str): Root directory path.
        img_size (int, optional): Image size for resizing. Default is 224.

    Returns:
        dict: A dictionary containing the image, text, and output label.

    """

    def __init__(self, mapping, data, dir_root_path, img_size=224):
        self.mapping = mapping
        self.data = data
        self.dir_root_path = dir_root_path
        self.img_size = img_size
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        # Retrieve person ID and label name from the mapping
        person_id = self.mapping.loc[idx, 'person']
        label_name = self.mapping.loc[idx, 'label']

        # Filter user tweets based on person ID and label name
        user_tweets = self.data[(self.data["person"] == person_id) & (self.data["label"] == label_name)]
        user_tweets = user_tweets.reset_index(drop=True)

        user_text = []
        user_img = []
        for i in range(user_tweets.shape[0]):
            img_path = user_tweets.loc[i, 'img_path']

            # Account for exceptions where there are no images (FILENOTFOUND) or corrupted at the path.
            try:
                path_to_img = self.dir_root_path + label_name + '/' + str(person_id) + '/images/' + img_path
                image = Image.open(path_to_img).convert('RGB')

            except:
                # If image is not found or corrupted, create a random image
                image = create_random_image(self.img_size)

            # Apply transforms to the image
            image = self.transform(image)
            user_img.append(image)

            text = user_tweets.loc[i, 'text']
            text = clean_text(text)
            user_text.append(text)

        # Get the unique label index for the output
        output = self.data['label'].unique().tolist().index(label_name)

        # Create a dictionary containing the image, text, and output label
        data = {'img': user_img[:32], 'text': user_text[:32], 'output': output}

        return data


def create_random_image(dim):
    '''
    Creates a random image of given dimensions.

    Args:
        dim (int): The dimension of the image (height and width).

    Returns:
        image (PIL.Image.Image): Randomly generated image.
    '''

    image = np.random.rand(dim, dim, 3)
    image = (255 * image).astype(np.uint8)
    image = Image.fromarray(image)

    return image


def clean_text(text):
    '''
    Cleans the input text by removing unwanted patterns and characters.

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.

    '''
    # Convert the input text to a string
    text = str(text)

    # Remove text within square brackets
    text = re.sub('\[.*?\]', '', text)

    # Remove URLs or website addresses
    text = re.sub('https?://\S+|www\.\S+', '', text)

    # Remove newline characters
    text = re.sub('\n', '', text)

    # Return the cleaned text
    return str(text)



#--------------------------- Hateful Memes -------------------------#


class HatefulMemesDataset(Dataset):
    def __init__(self, path, dataloader_type, batch_size, shuffle=True, cache_size=1000, data_filepath = None):
        self.path = path
        self.dataloader_type = dataloader_type
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cache_size = cache_size

        self.img_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomResizedCrop(size=(224, 224)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.data = self.load_data_from_file(filepath = data_filepath) if data_filepath else self.load_data()
        self.cache = {}


    def load_data(self):
        dataset_dict = load_dataset("neuralcatcher/hateful_memes")
        data = pd.DataFrame(dataset_dict[self.dataloader_type])
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    def load_data_from_file(self, filepath):
        data = self.read_jsonl_file_to_dataframe(filepath)
        if self.shuffle:
            data = data.sample(frac=1).reset_index(drop=True)
        return data

    def read_jsonl_file_to_dataframe(self, filepath):
        # Read the JSON objects from the file into a list
        with open(filepath) as f:
            json_objs = [json.loads(line) for line in f]

        # Convert the list of JSON objects into a DataFrame
        df = pd.DataFrame(json_objs)
        return df


    def __len__(self):
        return len(self.data) // self.batch_size

    
    def __getitem__(self, index):
        if index in self.cache:
            batch_dict = self.cache[index]
        else:
            # Get a new batch of data
            data_batch = self.data.iloc[index * self.batch_size : (index + 1) * self.batch_size, :]

            img_batch = np.array(data_batch['img'])
            text_batch = list(data_batch['text'])
            output_batch = np.array(data_batch['label']).reshape((self.batch_size,-1))
            img_batch = []

            for i, img_path in enumerate(data_batch['img']):
                img = cv2.imread(self.path + img_path)
                img = self.img_transforms(img)
                img = np.array(img).transpose((2,0,1))
                img_batch.append(img)

            batch_dict = {}
            batch_dict["img"] = img_batch
            batch_dict["text"] = text_batch
            batch_dict["output"] = output_batch


            if len(self.cache) == self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[index] = batch_dict

        return batch_dict















