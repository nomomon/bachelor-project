import numpy as np
from tqdm import tqdm
import torch

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

def run_train_epoch(epoch):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        
        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        logits = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )

        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(logits, batch.y))
        loss.backward()

        # Update using the gradients
        optimizer.step()

        # Apply argmax to get the predicted class
        all_preds.append(np.argmax(logits.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, 'train')
    return loss

def run_test_epoch(epoch):
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch.to(device)
        pred = model(
            batch.x.float(),
            batch.edge_index,
            batch.batch
        )
        loss = torch.sqrt(loss_fn(pred, batch.y))
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, 'test')
    return loss


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
        "model_dense_neurons": 64,
    }

    # Load the data
    train_dataset = DepressionDataset(root='data/', filename='train.tsv')
    test_dataset = DepressionDataset(root='data/', filename='dev.tsv', prefix="dev")

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

    # The class weights are imbalance, so we need to weight the loss function
    weights = compute_class_weight(train_dataset.get_targets())
    weights = torch.tensor(weights).to(device)
    print("Class weights:", weights)

    # Initialize the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    # Run the training loop
    N_EPOCHS = 6

    with mlflow.start_run() as run:
        for epoch in range(N_EPOCHS):
            # Training
            print(bcolors.BOLD, f"Epoch {epoch} | TRAIN", bcolors.ENDC, bcolors.OKBLUE)
            model.train()
            loss = run_train_epoch(epoch=epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Train Loss {loss}", bcolors.ENDC)
            mlflow.log_metric(key='Train loss', value=float(loss), step=epoch)

            # Testing
            model.eval()
            if epoch % 5 == 0:
                print(bcolors.BOLD, f"Epoch {epoch} | TEST", bcolors.ENDC, bcolors.OKGREEN)
                loss = run_test_epoch(epoch=epoch)
                loss = loss.detach().cpu().numpy()
                print(f"Test Loss {loss}", bcolors.ENDC)
                mlflow.log_metric(key='Test loss', value=float(loss), step=epoch)
            scheduler.step()
        
        # Save the model
        mlflow.pytorch.log_model(model, 'model')
        print('Done.')