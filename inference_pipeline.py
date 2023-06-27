

config_filename = 'late_fusion_user.yaml'

# export PYTHONPATH=$PYTHONPATH:`pwd`
#------------------------- Import Libraries -----------------------------------#

from dataset.dataloader import load_inference_data, load_user_inference_data
from models.train import train, test

import os
import torch
import pandas as pd
from torch import nn
import yaml
import importlib
import os
import torch

#-------------------- Initialize Parameters ----------------------------#

home_path = os.getcwd()

# Load the YAML file
with open(home_path + '/config/inference/' + config_filename) as f:
    configuration = yaml.safe_load(f)

configuration['General']['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

base_import_package = importlib.import_module(configuration['Models']['base_import_package'])

# Set the paths for the best model and metrics
configuration['Dataset']['best_model_path'] = home_path + configuration['Dataset']['best_model_path']
configuration['Dataset']['train_metrics_path'] = home_path + configuration['Dataset']['train_metrics_path']
configuration['Dataset']['test_metrics_path'] = home_path + configuration['Dataset']['test_metrics_path']

# Instantiate the image processor
configuration['Models']['image_processor'] = getattr(base_import_package, configuration['Models']['image_processor_package'])
configuration['Models']['image_processor'] = configuration['Models']['image_processor'].from_pretrained(configuration['Models']['image_processor_pretrained'])

# Instantiate the image model
configuration['Models']['image_model'] = getattr(base_import_package, configuration['Models']['image_model_package'])
configuration['Models']['image_model'] = configuration['Models']['image_model'].from_pretrained(configuration['Models']['image_model_name'])

# Instantiate the text tokenizer
configuration['Models']['text_tokenizer'] = getattr(base_import_package, configuration['Models']['text_tokenizer_package'])
configuration['Models']['text_tokenizer'] = configuration['Models']['text_tokenizer'].from_pretrained(configuration['Models']['text_processor_pretrained'])

# Instantiate the text model
configuration['Models']['text_model'] = getattr(base_import_package, configuration['Models']['text_model_package'])
configuration['Models']['text_model'] = configuration['Models']['text_model'].from_pretrained(configuration['Models']['text_model_name'])

print(f"\n*********** {configuration['Models']['model_class_name']} ***************\n")

#------------------- Dataloader -------------------#

data_path = configuration['Dataset']['data_path']
dir_root_path = configuration['Dataset']['dir_root_path']
IMG_SIZE = configuration['General']['img_size']
device = configuration['General']['device']

# Load data and create dataloaders
if configuration['General']['user']:
    # Load user inference data
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_user_inference_data(
        data_path, dir_root_path, IMG_SIZE, batch_size=configuration['General']['batch_size'])
else:
    # Load regular inference data
    train_dataloader, val_dataloader, test_dataloader, num_classes = load_inference_data(
        data_path, dir_root_path, IMG_SIZE, batch_size=configuration['General']['batch_size'])

#----------------- Initialize Encoders ------------------------#

# Image Models
img_processor = configuration['Models']['image_processor']
model_img = configuration['Models']['image_model'].to(device)

# Text Models
tokenizer_txt = configuration['Models']['text_tokenizer']
model_txt = configuration['Models']['text_model'].to(device)

# Import the module dynamically
module = importlib.import_module("models.fusion_models")

# Get the class object using getattr()
model = getattr(module, configuration['Models']['model_class_name'])
model = model(modality1 = model_img,modality2 = model_txt,configuration = configuration,device = device).to(device)


#--------------------- Train -----------------------------#

if configuration['General']['train']:
    print("\nTraining:")

    try:
        print("Loading Model")
        state_dict = torch.load(configuration['Dataset']['best_model_path'])
        model.load_state_dict(state_dict)
        print("Model Loaded")
    except:
        pass

    # Set up criterion and optimizer
    criterion = getattr(nn, configuration['Loss']['loss_fn'])().to(device)
    optimizer = getattr(torch.optim, configuration['Optimizers']['optimizer'])

    optimizer = optimizer([p for p in model.parameters() if p.requires_grad], lr=configuration['Optimizers']['learning_rate'])

    # Start training
    train(train_dataloader, val_dataloader, configuration, device, model, criterion, optimizer)

    print('Finished Training')

#-------------------------------- Test ------------------------------#

if configuration['General']['test']:
    print("\nTesting:")

    # Load the model
    state_dict = torch.load(configuration['Dataset']['best_model_path'])
    model.load_state_dict(state_dict)
    print("Model Loaded")

    # Perform testing
    test(test_dataloader, configuration, device, model)

    print('Finished Testing')





























