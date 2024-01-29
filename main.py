from pandas import json_normalize
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

import torch
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from torch_geometric.loader import DataLoader, ImbalancedSampler

from sklearn.metrics import (
    classification_report,
    confusion_matrix as sk_cm
)

from src.features.dataset import DepressionDataset
from src.models.GAT import GAT
from src.utils import clear_terminal

def get_metrics(target, prediction, set_type):
    report = classification_report(target, prediction, output_dict=True)
    report = json_normalize(report)
    report.columns = [f"{set_type}.{c}" for c in report.columns]
    report = report.iloc[0].to_dict()
    return report

def plot_cm(cm_s, epoch, root="./reports/figures"):
    fig, axs = plt.subplots(1, len(cm_s), figsize=(5 * len(cm_s), 5))

    for i, [cm, label] in enumerate(cm_s):
        ax = axs[i] if len(cm_s) > 1 else axs
            
        sns.heatmap(cm, annot=True, fmt='.2%', ax=ax, cbar=False, vmin=0, vmax=1)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(label)
        ax.set_aspect('equal')

    fig.suptitle(f'Confusion matrix at epoch {epoch}')

    plt.tight_layout()

    try:
        path = f'{root}/cm_{epoch:03d}.png'
        plt.savefig(path)
        plt.close()
    except:
        pass

    return path

def run_epoch(model, loader, optimizer, device, epoch, set_type):
    y_true = []
    y_pred = []
    epoch_loss = 0

    if set_type == 'train':
        desc = f'Epoch {epoch:3d} ┬ Train'
    elif set_type == 'valid':
        desc = '          └ Valid'

    for batch in tqdm(loader, desc=desc):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            batch.x,
            batch.edge_index,
            batch.batch
        )
        loss = corn_loss(out, batch.y, num_classes=3)
        
        if set_type == 'train':
            loss.backward()
            optimizer.step()

        pred = corn_label_from_logits(out)

        y_true += batch.y.cpu().tolist()
        y_pred += pred.cpu().tolist()
        epoch_loss += loss.item()
    epoch_loss /= len(loader)

    return y_true, y_pred, epoch_loss

def main(params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_set = DepressionDataset('train', params["encoder_type"], params["graph_type"])
    valid_set = DepressionDataset('valid', params["encoder_type"], params["graph_type"])

    if False:
        train_set.process()
        valid_set.process()

    # Get sampler
    print("Getting sampler...")
    sampler = ImbalancedSampler(train_set, num_samples=len(train_set))

    # Get loaders
    print("Getting loaders...")
    train_loader = DataLoader(train_set, batch_size=params["batch_size"], sampler=sampler)
    valid_loader = DataLoader(valid_set, batch_size=params["batch_size"], shuffle=False)

    # Get model
    print("Getting model...")
    model = GAT(
        feature_size=train_set[0].x.shape[1], # 300 or 768
        model_params=params,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    early_stopping_counter = 0
    min_valid_loss = np.inf

    # Start training
    with mlflow.start_run():
        mlflow.log_params(params)

        print("Start training...")
        for epoch in range(0, 50):
            # Train
            model.train()
            y_true, y_pred, train_loss = run_epoch(model, train_loader, optimizer, device, epoch, 'train')
            train_cm = sk_cm(y_true, y_pred, labels=[0, 1, 2], normalize='true')
            mlflow.log_metrics(get_metrics(y_true, y_pred, "train"), step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)

            # Valid
            model.eval()
            y_true, y_pred, valid_loss = run_epoch(model, valid_loader, optimizer, device, epoch, 'valid')
            valid_cm = sk_cm(y_true, y_pred, labels=[0, 1, 2], normalize='true')
            mlflow.log_metrics(get_metrics(y_true, y_pred, "valid"), step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

            # Plot confusion matrix
            cm_path = plot_cm([
                [train_cm, "Train"],
                [valid_cm, "Valid"]
            ], epoch) 
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

            # Early stopping
            if epoch > 0 and valid_loss > min_valid_loss:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                min_valid_loss = valid_loss

            if early_stopping_counter >= 3:
                print('Early stopping...')
                mlflow.log_metric("early_stopping", 1, step=epoch)
                break
        
        if early_stopping_counter < 3:
            mlflow.log_metric("early_stopping", 0, step=epoch)

        print('Done!')
        clear_terminal()
    
    return min_valid_loss