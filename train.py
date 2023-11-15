import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from torch_geometric.nn import summary

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
    # roc_auc_score
)
import mlflow.pytorch
from utils import bcolors, compute_class_weight

from torch_geometric.loader import DataLoader
from dataset import DepressionDataset
from model import GNN

# TODO: generally speaking, what's up with this code clean this up
#%%
def log_conf_matrix(y_pred, y_true, epoch):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["not depression", "moderate", "severe"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/cm_{epoch}.png')
    mlflow.log_artifact(f"data/images/cm_{epoch}.png")

# TODO: convert train one epoch and run test epoch to same function
#       but it has a flag for backward pass or for train or not

def train_one_epoch(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    epoch_loss = 0
    steps_per_epoch = 0 # TODO: I think we can use len(train_loader) instead

    for batch in tqdm(train_loader):
        steps_per_epoch += 1

        # Use GPU
        batch.to(device)
        
        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        logits = model(
            batch.x.float(),    # node features
            batch.edge_index,   # adj list
            batch.batch         # batch info    
        )

        # Calculating the loss and gradients
        loss = loss_fn(logits, batch.y)
        loss.backward()

        # Update using the gradients
        optimizer.step()

        # Update tracking
        epoch_loss += loss.item()

        # Apply argmax to get the predicted class
        logits_np = logits.cpu().detach().numpy()
        labels_np = batch.y.cpu().detach().numpy()
        all_preds.append(np.argmax(logits_np, axis=1))
        all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    calculate_metrics(all_preds, all_labels, epoch, 'train')
    return epoch_loss / steps_per_epoch

def run_test_epoch(epoch, model, test_loader, loss_fn):
    all_preds = []
    all_labels = []
    epoch_loss = 0
    steps_per_epoch = 0

    for batch in test_loader:
        steps_per_epoch += 1

        batch.to(device)
        pred = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )
        loss = loss_fn(pred, batch.y)

        # Update tracking
        epoch_loss += loss.item()

        # Apply argmax to get the predicted class
        logits_np = pred.cpu().detach().numpy()
        labels_np = batch.y.cpu().detach().numpy()
        all_preds.append(np.argmax(logits_np, axis=1))
        all_labels.append(labels_np)

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    calculate_metrics(all_preds, all_labels, epoch, 'test')
    return epoch_loss / steps_per_epoch


# export this to utils.py
def calculate_metrics(y_pred, y_true, epoch, type):

    conf_mat = confusion_matrix(y_pred, y_true)
    conf_mat = '\t'+str(conf_mat).replace('\n', '\n\t')

    print(f"""
    Accuracy: {accuracy_score(y_pred, y_true)}
    Confusion matrix: \n {conf_mat}
    Micro Precision: {precision_score(y_pred, y_true, average='micro')}
    Micro Recall: {recall_score(y_pred, y_true, average='micro')}
    Micro F1 Score: {f1_score(y_pred, y_true, average='micro')}
    """, end="")

    mlflow.log_metric(f"{type}_micro_f1", f1_score(y_pred, y_true, average='micro'), step=epoch)
    mlflow.log_metric(f"{type}_accuracy", accuracy_score(y_pred, y_true), step=epoch)
    mlflow.log_metric(f"{type}_micro_precision", precision_score(y_pred, y_true, average='micro'), step=epoch)
    mlflow.log_metric(f"{type}_micro_recall", recall_score(y_pred, y_true, average='micro'), step=epoch)

    # TODO: add the confusion matrix to the mlflow log
    # TODO: check with the paper if they used micro or macro



if __name__ == "__main__":
    # os.system('rm -rf data/processed')

    # Model hyperparameters
    # TODO: find these hyperparameters
    # TODO: add these hyperparameters to the mlflow log
    model_params = {
        "model_embedding_size": 32,
        "model_attention_heads": 4,
        "model_layers": 3,
        "model_dropout_rate": 0.1,
        "model_top_k_ratio": 0.5,
        "model_dense_neurons": 32,
        # "data_path": "fully_connected_graph"
    }

    # Load the data
    train_dataset = DepressionDataset(
        root='data/', filename='train_oversampled.tsv', prefix="train")
    test_dataset = DepressionDataset(
        root='data/', filename='dev.tsv', prefix="dev"
    )

    # Create the data loaders
    NUM_GRAPHS_PER_BATCH = 256
    train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)

    # Use GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = GNN(
        feature_size=train_dataset[0].x.shape[1],
        num_classes=3,
        model_params=model_params
    )
    model = model.to(device)
    print("Number of parameters:", model.count_parameters())
    print("Model summary:")
    print(summary(
        model, train_dataset[0].x, 
        train_dataset[0].edge_index, 
        batch_index=train_dataset[0].batch,
        max_depth=1))

    # The class weights are imbalance, so we need to weight the loss function
    weights = compute_class_weight(train_dataset.get_targets())
    weights = torch.tensor(weights).to(device)
    print("Class weights:", weights)

    # Initialize the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.8, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Run the training loop
    # TODO: add early stopping
    N_EPOCHS = 500

    with mlflow.start_run() as run:
        # log the params used in the experiment
        mlflow.log_params(model_params)

        for epoch in range(N_EPOCHS):
            # Training
            print(bcolors.BOLD, f"Epoch {epoch} | TRAIN", bcolors.ENDC, bcolors.OKBLUE)
            model.train()

            loss = train_one_epoch(epoch, model, train_loader, optimizer, loss_fn)
            
            print(f"Train Loss {loss}", bcolors.ENDC)
            mlflow.log_metric(key='train_loss', value=float(loss), step=epoch)

            # Testing
            model.eval()
            if epoch % 5 == 0:
                print(bcolors.BOLD, f"Epoch {epoch} | TEST", bcolors.ENDC, bcolors.OKGREEN)
                
                loss = run_test_epoch(epoch, model, test_loader, loss_fn)

                print(f"Test Loss {loss}", bcolors.ENDC)
                mlflow.log_metric(key='test_loss', value=float(loss), step=epoch)
            scheduler.step()
        
        # Save the model
        mlflow.pytorch.log_model(model, 'model')
        print('Done.')