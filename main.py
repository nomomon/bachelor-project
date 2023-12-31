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
    f1_score,
    accuracy_score,
    confusion_matrix as sk_cm
)

from src.features.dataset import DepressionDataset
from src.models.GAT import GAT
from src.utils import clear_terminal

def get_metrics(y_true, y_pred, set_type):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    res = {
        f'{set_type}_acc': acc,
        f'{set_type}_f1': f1,
    }

    return res

def plot_cm(train_cm, valid_cm, epoch):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    sns.heatmap(train_cm, annot=True, fmt='.2%', ax=axs[0], cbar=False, vmin=0, vmax=1)
    axs[0].set_xlabel('Predicted')
    axs[0].set_ylabel('Actual')
    axs[0].set_title('Train')
    axs[0].set_aspect('equal')

    sns.heatmap(valid_cm, annot=True, fmt='.2%', ax=axs[1], cbar=False, vmin=0, vmax=1)
    axs[1].set_xlabel('Predicted')
    axs[1].set_ylabel('Actual')
    axs[1].set_title('Valid')
    axs[1].set_aspect('equal')

    fig.suptitle(f'Confusion matrix at epoch {epoch}')

    plt.tight_layout()

    path = f'./reports/figures/cm_{epoch:03d}.png'
    plt.savefig(path)
    plt.close()

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
            cm_path = plot_cm(train_cm, valid_cm, epoch)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrix")

            # Early stopping
            if epoch > 0 and valid_loss > min_valid_loss:
                early_stopping_counter += 1
            else:
                early_stopping_counter = 0
                min_valid_loss = valid_loss

            if early_stopping_counter >= 10:
                print('Early stopping...')
                mlflow.log_metric("early_stopping", 1, step=epoch)
                break
        
        if early_stopping_counter < 10:
            mlflow.log_metric("early_stopping", 0, step=epoch)

        print('Done!')
        clear_terminal()
    
    return min_valid_loss

if __name__ == '__main__':

    param_space = dict(
        encoder_type = ['bert', 'w2v'],
        graph_type = ['window', 'dependency'],
        lr = [1e-3, 1e-4, 1e-5],
        weight_decay = [1e-2, 1e-3, 1e-4, 1e-5],
        batch_size = [32, 64, 128, 256],
        model__layers = [3],
        model__dense_neurons = [32, 64, 128],
        model__embedding_size = [32, 64, 128],
        model__num_classes = [3 - 1], # -1 because of coral
        model__n_heads = [1, 3, 5],
        model__dropout = [0.1, 0.3, 0.5],
    )

    # random search
    while True:
        params = {}
        for k, v in param_space.items():
            params[k] = v[np.random.randint(len(v))]
        main(params)