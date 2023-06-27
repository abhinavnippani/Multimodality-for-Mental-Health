from tqdm import tqdm
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score


#------------------------ Train ---------------------------------#

def train(train_dataloader, val_dataloader, configuration, device, model, criterion, optimizer, trial=None):
    """
    Function to train the model.

    Args:
        train_dataloader (DataLoader): DataLoader for the training dataset.
        val_dataloader (DataLoader): DataLoader for the validation dataset.
        configuration (dict): Configuration dictionary.
        device (torch.device): Device on which the model and data tensors are located.
        model (nn.Module): Model to be trained.
        criterion: Loss criterion.
        optimizer: Optimizer for training the model.
        trial (optuna.trial.Trial, optional): Trial object for optuna hyperparameter optimization. Defaults to None.

    Returns:
        None

    """

    # Image processor
    img_processor = configuration['Models']['image_processor']
    # Text tokenizer
    tokenizer_txt = configuration['Models']['text_tokenizer']

    try:
        train_metrics_df_og = pd.read_csv(configuration['Dataset']['train_metrics_path'])
    except:
        train_metrics_df_og = pd.DataFrame(columns=["epoch", "num_steps", "train_loss", "val_loss", "train_acc", "val_acc", "train_precision", "val_precision", "train_recall", "val_recall", "train_f1score", "val_f1score", "train_aucroc", "val_aucroc"])

    if train_metrics_df_og.shape[0] > 0:
        best_val_f1 = train_metrics_df_og["val_f1score"].max()
        print(best_val_f1)
    else:
        best_val_f1 = 0

    train_metrics = {"epoch": [], "num_steps": [], "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_precision": [], "val_precision": [], "train_recall": [], "val_recall": [], "train_f1score": [], "val_f1score": [], "train_aucroc": [], "val_aucroc": []}

    for epoch in range(train_metrics_df_og.shape[0] + 1, configuration['General']['epochs']+1):

        num_steps = 0
        val_num_steps = 0
        running_loss = 0.0
        val_running_loss = 0.0
        true = []
        pred = []
        val_true = []
        val_pred = []

        # ----------------------------------- Train -----------------------------#

        model.train()

        for data in tqdm(train_dataloader, desc=f"[Epoch {epoch}]", ascii=' >='):

            img_batch = data['img']
            text_batch = data['text']
            labels = data['output']
            labels_one_hot = F.one_hot(labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])

            if configuration['General']['user']:
                outputs = model(img_batch, text_batch).to(device)
                labels_one_hot = labels_one_hot.unsqueeze(0)
            else:
                text_feats = tokenizer_txt(text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                img_processed = img_processor(img_batch, return_tensors='pt').to(device)

                outputs = model(img_processed, text_feats).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            if configuration['Models']['regularization']:
                regularization_loss = 0
                for param in model.parameters():
                    regularization_loss += torch.sum(torch.square(param))  # L2 regularization

                loss = criterion(outputs, labels_one_hot.to(torch.float32).to(device)) + configuration['Loss']['reg_lambda'] * regularization_loss
            else:
                loss = criterion(outputs, labels_one_hot.to(torch.float32).to(device))

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            true.extend(labels_one_hot.cpu().detach().numpy())
            pred.extend(outputs.cpu().detach().numpy())

            num_steps += 1

        # ----------------------------------- Validation -----------------------------#

        model.eval()

        # Validation
        with torch.no_grad():
            for val_data in tqdm(val_dataloader, desc=f"[Epoch {epoch}]", ascii=' >='):

                val_img_batch = val_data['img']
                val_text_batch = val_data['text']

                val_labels = val_data['output']
                val_labels_one_hot = F.one_hot(val_labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])

                if configuration['General']['user']:
                    val_outputs = model(val_img_batch, val_text_batch).to(device)
                    val_labels_one_hot = val_labels_one_hot.unsqueeze(0)
                else:
                    text_feats = tokenizer_txt(val_text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
                    img_processed = img_processor(val_img_batch, return_tensors='pt').to(device)

                    val_outputs = model(img_processed, text_feats).to(device)

                val_loss = criterion(val_outputs, val_labels_one_hot.to(torch.float32).to(device))
                val_running_loss += val_loss.item()

                val_true.extend(val_labels_one_hot.cpu().detach().numpy())
                val_pred.extend(val_outputs.cpu().detach().numpy())

                val_num_steps += 1

        # ----------------------------------- Metrics -----------------------------#

        train_acc, train_precision, train_recall, train_f1, train_roc_auc = get_metrics(true, pred)
        val_acc, val_precision, val_recall, val_f1, val_roc_auc = get_metrics(val_true, val_pred)

        train_metrics["epoch"].append(epoch)
        train_metrics["num_steps"].append(num_steps)
        train_metrics["train_loss"].append(running_loss/num_steps)
        train_metrics["val_loss"].append(val_running_loss/val_num_steps)
        train_metrics["train_acc"].append(train_acc)
        train_metrics["val_acc"].append(val_acc)
        train_metrics["train_precision"].append(train_precision)
        train_metrics["val_precision"].append(val_precision)
        train_metrics["train_recall"].append(train_recall)
        train_metrics["val_recall"].append(val_recall)
        train_metrics["train_f1score"].append(train_f1)
        train_metrics["val_f1score"].append(val_f1)
        train_metrics["train_aucroc"].append(train_roc_auc)
        train_metrics["val_aucroc"].append(val_roc_auc)

        print(f'Num_steps : {num_steps}, train_loss : {running_loss/num_steps:.3f}, val_loss : {val_running_loss/val_num_steps:.3f}, train_acc : {train_acc}, val_acc : {val_acc}, train_f1score : {train_f1}, val_f1score : {val_f1}, train_aucroc : {train_roc_auc}, val_aucroc : {val_roc_auc}')

        train_metrics_df = pd.DataFrame(train_metrics)
        train_metrics_df = pd.concat([train_metrics_df_og, train_metrics_df], axis=0, ignore_index=True).reset_index(drop=True)

        train_metrics_df.to_csv(configuration['Dataset']['train_metrics_path'], index=False)

        # ----------------------------------- Save Model -----------------------------#

        if train_metrics["val_f1score"][-1] >= best_val_f1:
            best_val_f1 = train_metrics["val_f1score"][-1]
            best_model = model

            print("Saving the model")

            torch.save(model.state_dict(), configuration['Dataset']['best_model_path'])

        # # Early Stopping trials
        # if(trial):
        #     trial.report(val_acc, epoch)
        #
        #     if trial.should_prune():
        #         raise optuna.exceptions.TrialPruned()

#-------------------------- Test -------------------------------#

def test(test_dataloader, configuration, device, model):
    """
    Perform testing on the test dataset using the trained model and compute evaluation metrics.

    Args:
        test_dataloader (DataLoader): DataLoader for the test dataset.
        configuration (dict): Configuration parameters.
        device (torch.device): Device to perform the computations on (e.g., 'cuda', 'cpu').
        model (nn.Module): Trained model.

    Returns:
        None
    """

    # Image processor
    img_processor = configuration['Models']['image_processor']
    # Text tokenizer
    tokenizer_txt = configuration['Models']['text_tokenizer']

    test_true = []
    test_pred = []

    test_metrics = {"test_acc": [], "test_precision": [], "test_recall": [], "test_f1score": [], "test_aucroc": []}

    for test_data in tqdm(test_dataloader, ascii=' >='):
        test_img_batch = test_data['img']
        test_text_batch = test_data['text']
        test_labels = test_data['output']

        test_labels_one_hot = F.one_hot(test_labels.to(torch.int64).squeeze(), configuration['Models']['mlp_num_classes'])

        if configuration['General']['user']:
            test_outputs = model(test_img_batch, test_text_batch).to(device)
            test_labels_one_hot = test_labels_one_hot.unsqueeze(0)
        else:
            text_feats = tokenizer_txt(test_text_batch, return_tensors="pt", padding=True, truncation=True).to(device)
            img_processed = img_processor(test_img_batch, return_tensors='pt').to(device)

            test_outputs = model(img_processed, text_feats).to(device)

        test_true.extend(test_labels_one_hot.cpu().detach().numpy())
        test_pred.extend(test_outputs.cpu().detach().numpy())

    # Compute evaluation metrics
    test_acc, test_precision, test_recall, test_f1, test_roc_auc = get_metrics(test_true, test_pred)

    test_metrics["test_acc"].append(test_acc)
    test_metrics["test_precision"].append(test_precision)
    test_metrics["test_recall"].append(test_recall)
    test_metrics["test_f1score"].append(test_f1)
    test_metrics["test_aucroc"].append(test_roc_auc)

    print(test_metrics)

    # Save test metrics to a CSV file
    test_metrics_df = pd.DataFrame(test_metrics)
    test_metrics_df.to_csv(configuration['Dataset']['test_metrics_path'], index=False)


#-------------------------- Metrics --------------------------#

def get_metrics(true, pred):
    """
    Compute evaluation metrics based on true labels and predicted labels.

    Args:
        true (numpy.ndarray): Array of true labels.
        pred (numpy.ndarray): Array of predicted labels.

    Returns:
        tuple: Tuple containing the computed metrics: accuracy, precision, recall, F1 score, and AUC-ROC.
    """

    # Compute accuracy
    acc = accuracy_score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    # Compute precision
    precision = precision_score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    # Compute recall
    recall = recall_score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    # Compute F1 score
    f1 = f1_score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    # Compute AUC-ROC
    roc_auc = roc_auc_score(np.argmax(true, axis=1), np.argmax(pred, axis=1))

    return acc, precision, recall, f1, roc_auc
