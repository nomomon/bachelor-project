import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch

from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
import mlflow.pytorch

import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def log_conf_matrix(y_pred, y_true, epoch, step_type):
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    
    # Create a DataFrame for confusion matrix
    classes = ["not depression", "moderate", "severe"]
    df_cfm = pd.DataFrame(cm, index=classes, columns=classes)
    
    # Plot and save confusion matrix as an image
    plt.figure(figsize=(10, 7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    image_path = f'data/images/cm_{epoch:04d}_{step_type}.png'
    cfm_plot.figure.savefig(image_path)
    
    # Log confusion matrix image as an artifact
    mlflow.log_artifact(image_path)

def log_metrics(y_pred, y_true, epoch, metric_type):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    macro_precision = precision_score(y_true, y_pred, average='macro')
    macro_recall = recall_score(y_true, y_pred, average='macro')

    # Log metrics
    metrics = {
        f"{metric_type}_accuracy": accuracy,
        f"{metric_type}_macro_f1": macro_f1,
        f"{metric_type}_macro_precision": macro_precision,
        f"{metric_type}_macro_recall": macro_recall
    }
    mlflow.log_metrics(metrics, step=epoch)

def step(epoch, model, data_loader, optimizer, loss_fn, device, step_type="train"):
    all_preds = []
    all_labels = []
    total_loss = 0
    total_steps = len(data_loader)

    for batch in tqdm(data_loader, total=total_steps):
        optimizer.zero_grad()
        batch.to(device)
        
        logits = model(
            batch.x.float(),    # node features
            batch.edge_index,   # adj list
            batch.batch         # batch info    
        )
        loss = loss_fn(logits, batch.y)

        if step_type == "train":
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

        # Apply argmax to get the predicted class
        logits_np = logits.cpu().detach().numpy()
        labels_np = batch.y.cpu().detach().numpy()
        all_preds.append(np.argmax(logits_np, axis=1))
        all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    log_metrics(all_preds, all_labels, epoch, step_type)
    log_conf_matrix(all_preds, all_labels, epoch, step_type)

    return total_loss / total_steps

def train(params, train_dataset, valid_dataset, model_class):
    with mlflow.start_run() as run:
        mlflow.log_params(params)

        train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"], shuffle=True)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        model_params = {k: v for k, v in params.items() if k.startswith("model_")}
        model = model_class(
            feature_size=train_dataset[0].x.shape[1], 
            model_params=model_params)
        model.to(device)
        mlflow.log_param("num_params", count_parameters(model))

        loss_fn = torch.nn.NLLLoss()
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=params["learning_rate"],
                                    momentum=params["sgd_momentum"],
                                    weight_decay=params["weight_decay"])
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                           gamma=params["scheduler_gamma"])

        best_valid_loss = 1e6
        early_stopping_counter = 0
        for epoch in range(params["n_epochs"]):
            model.train()

            train_loss = step(epoch, model, train_loader, optimizer, loss_fn, device)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            if epoch % 5 == 0:
                model.eval()
                valid_loss = step(epoch, model, valid_loader, optimizer, loss_fn, device, step_type="valid")
                mlflow.log_metric("valid_loss", valid_loss, step=epoch)

                if best_valid_loss > valid_loss:
                    best_valid_loss = valid_loss
                    mlflow.pytorch.log_model(model, 'model')
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
            scheduler.step()

            if early_stopping_counter >= 10:
                print("Early stopping due to no improvement.")
                break
    return [best_valid_loss]
        
        






