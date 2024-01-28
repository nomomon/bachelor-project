from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import json

import torch
from torch_geometric.loader import DataLoader, ImbalancedSampler

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    confusion_matrix as sk_cm
)

from bayes_opt import BayesianOptimization

from src.features.dataset import DepressionDataset
from src.models.GAT import GAT
from src.utils import clear_terminal

def get_metrics(y_true, y_pred, set_type):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')

    res = {
        f'{set_type}_acc': acc,
        f'{set_type}_f1': f1,
        f'{set_type}_precision': precision,
        f'{set_type}_recall': recall,
    }

    return res

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
        loss = torch.nn.functional.cross_entropy(out, batch.y)
        
        if set_type == 'train':
            loss.backward()
            optimizer.step()

        pred = out.argmax(dim=1)

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
        mlflow.log_param("loss_func", "cross_entropy")

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

def main_fixed(**params):
    new_params = {
        "encoder_type": "bert" if params["encoder_type"] > 0.5 else "w2v",
        "graph_type": "dependency" if params["graph_type"] > 0.5 else "window",
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": nearest_pow2(params["batch_size"]),
        "model__layers": 3,
        "model__dense_neurons": nearest_pow2(params["model__dense_neurons"]),
        "model__embedding_size": nearest_pow2(params["model__embedding_size"]),
        "model__num_classes": 3,
        "model__n_heads": int(params["model__n_heads"]),
        "model__dropout": params["model__dropout"],
    }

    return - main(new_params)

def nearest_pow2(x):
    x = int(x)
    return 2 ** int(np.log2(x))

def load_prev_runs(optimizer):
    # get the latest runs from mlflow
    runs = mlflow.search_runs(
        experiment_ids="0",
        order_by=["attributes.start_time desc"],
        max_results=500,
    )

    # load the parameters from the runs
    for _, row in runs.iterrows():
        row_params = {
            k.split(".", 1)[1]: v for k, v in row.items() if k.startswith("params.")
        }
        params = {}

        params["encoder_type"] = 1 if row_params["encoder_type"] == "bert" else 0
        params["graph_type"] = 1 if row_params["graph_type"] == "dependency" else 0
        params["lr"] = float(row_params["lr"])
        params["weight_decay"] = float(row_params["weight_decay"])
        params["batch_size"] = int(row_params["batch_size"])
        params["model__dense_neurons"] = int(row_params["model__dense_neurons"])
        params["model__embedding_size"] = int(row_params["model__embedding_size"])
        params["model__n_heads"] = int(row_params["model__n_heads"])
        params["model__dropout"] = float(row_params["model__dropout"])

        optimizer.register(
            params=params,
            target=- float(row["metrics.valid_loss"]),
        )

if __name__ == '__main__' and False:
    # with open('./config.json', 'r') as f:
    #     config = json.load(f)

    # mlflow.set_tracking_uri(config["mlflow_uri"])

    param_space = dict(
        encoder_type = [0, 1],
        graph_type = [0, 1],
        lr = [1e-5, 1],
        weight_decay = [1e-5, 1],
        batch_size = [16, 256],
        model__dense_neurons = [16, 256],
        model__embedding_size = [16, 256],
        model__n_heads = [1, 5],
        model__dropout = [0.001, 0.5],
    )

    # bayesian optimization
    optimizer = BayesianOptimization(
        f=main_fixed,
        pbounds=param_space,
        random_state=1,
    )
    
    load_prev_runs(optimizer)    

    optimizer.maximize(
        n_iter=50
    )