import mlflow
import torch
from tqdm import tqdm
import numpy as np

from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from src.data.text_preprocessing import preprocess_text


from src.features.dataset import MyOwnDataset
from src.models.GCN import GCN


def log_metrics(y_pred, y_true, epoch, metric_type):
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

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

    for batch in tqdm(data_loader, desc=f"{step_type.capitalize()}ing epoch {epoch}"):
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

    return total_loss / total_steps


if __name__ == '__main__':
    params = {
        "batch_size": 32,
        "n_epochs": 100,
        "learning_rate": 0.01,
        "sgd_momentum": 0.9,
        "weight_decay": 1e-4,
        "scheduler_gamma": 0.95,
        "model_layers": 1,
        "model_dense_neurons": 8,
        "model_embedding_size": 32,
        "model_num_classes": 3,
        "model_dropout": 0.05,
    }

    train_dataset = MyOwnDataset('train', 'w2v', 'window', pre_transform=preprocess_text)
    valid_dataset = MyOwnDataset('valid', 'w2v', 'window', pre_transform=preprocess_text)

    samples_weight = torch.from_numpy(np.array([.4, .15, .45]))
    samples_weigth = samples_weight.double()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], sampler=sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=params["batch_size"])

    print("Num. mini-batches in train loader:", len(train_loader))
    print("Num. mini-batches in valid loader:", len(valid_loader))
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_params = {k: v for k, v in params.items() if k.startswith("model_")}
    model = GCN(
        feature_size=train_dataset[0].x.shape[1], 
        model_params=model_params)
    model.to(device)

    loss_fn = torch.nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=params["learning_rate"],
                                momentum=params["sgd_momentum"],
                                weight_decay=params["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                        gamma=params["scheduler_gamma"])

    best_valid_loss = 1e6
    early_stopping_counter = 0
    epoch_iter = tqdm(range(params["n_epochs"]))
    for epoch in epoch_iter:
        model.train()

        train_loss = step(epoch, model, train_loader, optimizer, loss_fn, device)
        mlflow.log_metric("train_loss", train_loss, step=epoch)

        if epoch % 5 == 0:
            model.eval()
            valid_loss = step(epoch, model, valid_loader, optimizer, loss_fn, device, step_type="valid")
            mlflow.log_metric("valid_loss", valid_loss, step=epoch)

            if best_valid_loss - valid_loss > 1e-4:
                best_valid_loss = valid_loss
                mlflow.pytorch.log_model(model, 'model')
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
            
        
        scheduler.step()

        epoch_iter.set_description(f"TL: {train_loss:.4f} | VL: {valid_loss:.4f} | EaStp: {early_stopping_counter}/10")
        
        if early_stopping_counter >= 10:
            print("Early stopping due to no improvement.")
            break
    
    print("Training finished.")
