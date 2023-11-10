# %% imports
import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import numpy as np
from tqdm import tqdm
import os
from dataset_featurizer import DepressionDataset
from model import GNN
import mlflow.pytorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# %% Delete the previous proccessed dataset
os.system("rm -rf data/processed")

# %% Loading the dataset
train_dataset = DepressionDataset(root="data/", filename="train.tsv")
test_dataset = DepressionDataset(root="data/", filename="dev.tsv", test=True)

# %% Loading the model
model = GNN(feature_size=train_dataset[0].x.shape[1])
model = model.to(device)
print(f"Number of parameters: {count_parameters(model)}")
model

# %% Loss and Optimizer
# use class weights to balance the classes
weights = torch.tensor([3.0, 1.0, 9.0]).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


# %% Prepare training
NUM_GRAPHS_PER_BATCH = 256
train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)


def train(epoch):
    # Enumerate over the data
    all_preds = []
    all_labels = []
    for _, batch in enumerate(tqdm(train_loader)):
        # Use GPU
        batch.to(device)
        # Reset gradients

        optimizer.zero_grad()
        # Passing the node features and the connection info
        pred = model(
            batch.x.float(),
            None,  # it needs to be edge attr
            batch.edge_index,
            batch.batch,
        )

        # Calculating the loss and gradients
        loss = torch.sqrt(loss_fn(pred, batch.y))
        loss.backward()
        # Update using the gradients
        optimizer.step()

        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "train")
    return loss


def test(epoch):
    all_preds = []
    all_labels = []
    for batch in test_loader:
        batch.to(device)
        pred = model(
            batch.x.float(),
            None,  # it needs to be edge attr
            batch.edge_index,
            batch.batch,
        )
        loss = torch.sqrt(loss_fn(pred, batch.y))
        all_preds.append(np.argmax(pred.cpu().detach().numpy(), axis=1))
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels, epoch, "test")
    return loss


def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"Micro F1 Score: {f1_score(y_pred, y_true, average='micro')}")
    print(f"Accuracy: {accuracy_score(y_pred, y_true)}")
    print(f"Micro Precision: {precision_score(y_pred, y_true, average='micro')}")
    print(f"Micro Recall: {recall_score(y_pred, y_true, average='micro')}")
    try:
        roc = roc_auc_score(y_pred, y_true)
        print(f"ROC AUC: {roc}")
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(roc), step=epoch)
    except:
        mlflow.log_metric(key=f"ROC-AUC-{type}", value=float(0), step=epoch)
        print(f"ROC AUC: notdefined")


# %% Run the training
train_losses = []
test_losses = []

with mlflow.start_run() as run:
    for epoch in range(500):
        # Training
        model.train()
        loss = train(epoch=epoch)
        loss = loss.detach().cpu().numpy()
        print(f"Epoch {epoch} | Train Loss {loss}")
        mlflow.log_metric(key="Train loss", value=float(loss), step=epoch)
        train_losses.append([epoch, loss])

        # Testing
        model.eval()
        if epoch % 5 == 0:
            loss = test(epoch=epoch)
            loss = loss.detach().cpu().numpy()
            print(f"Epoch {epoch} | Test Loss {loss}")
            mlflow.log_metric(key="Test loss", value=float(loss), step=epoch)
            test_losses.append([epoch, loss])

        scheduler.step()
    print("Done.")


# %% Save the model
mlflow.pytorch.log_model(model, "model")

# %% Make the train/validation loss plot
import matplotlib.pyplot as plt


def plot_loss(losses, title):
    plt.gca().plot([x[0] for x in losses], [x[1] for x in losses], title=title)


plt.figure(figsize=(10, 5))
plot_loss(train_losses, "Train loss")
plot_loss(test_losses, "Test loss")
plt.legend(["Train", "Test"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.savefig("loss.png")

# %%
