from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

import torch
from torch.optim.lr_scheduler import ExponentialLR
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from torch_geometric.loader import DataLoader, ImbalancedSampler

from mango import Tuner
from mango.domain.distribution import loguniform

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix
)

from src.features.dataset import DepressionDataset
from src.models.GAT import GAT
from src.utils import clear_terminal

def get_metrics(y_true, y_pred, set_type):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    roc_auc = roc_auc_score(y_true, y_pred, average='macro', multi_class='ovo')
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        f'{set_type}_acc': acc,
        f'{set_type}_f1_macro': f1,
        f'{set_type}_roc_auc_macro': roc_auc,
        f'{set_type}_cm': str(cm)
    }

def run_epoch(model, loader, optimizer, device, epoch, set_type):
    y_true = []
    y_pred = []
    epoch_loss = 0

    if set_type == 'train':
        desc = f'Epoch {epoch:2d} ┬ {set_type}'
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
    params = params[0]

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
    scheduler = ExponentialLR(optimizer, gamma=params["gamma"])

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
            mlflow.log_metrics(get_metrics(y_true, y_pred, "train"), step=epoch)
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            scheduler.step()

            # Valid
            model.eval()
            y_true, y_pred, valid_loss = run_epoch(model, valid_loader, optimizer, device, epoch, 'valid')
            mlflow.log_metrics(get_metrics(y_true, y_pred, "valid"), step=epoch)
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

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
        graph_type = ['dependency'],
        lr = loguniform(-5, 5),
        weight_decay = loguniform(-5, 5),
        gamma = loguniform(-5, 5),
        batch_size = [32, 64, 128, 256],
        model__layers = [3],
        model__dense_neurons = [16, 64, 128, 256],
        model__embedding_size = [16, 64, 128, 256],
        model__num_classes = [3 - 1], # -1 because of coral
        model__n_heads = [1, 3, 4, 5],
        model__dropout = [0.1, 0.2, 0.3, 0.4, 0.5],
    )

    tuner = Tuner(param_space, main)
    results = tuner.minimize()
    print(f'Optimal value of parameters: {results["best_params"]} and objective: {results["best_objective"]}')