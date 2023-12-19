import mlflow
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import WeightedRandomSampler
from torch_geometric.loader import DataLoader

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from src.data.utils import read_tsv

from src.features.dataset import DepressionDataset
from src.models.GCN import GCN


batch_size = 128
reset_dataset = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    train_set = DepressionDataset('train', 'bert', 'dependency')
    valid_set = DepressionDataset('valid', 'bert', 'dependency')

    if reset_dataset:
        train_set.process()
        valid_set.process()

    # # Get class weights
    # print("Getting class weights...", end=' ')
    # label_to_class = {"not depression": 0, "moderate": 1, "severe": 2}
    # labels = read_tsv('data/silver/train.tsv')["label"].apply(lambda x: label_to_class[x]).values
    # class_weights = 1 / np.bincount(labels)
    # class_weights = class_weights / class_weights.sum()
    # class_weights = torch.FloatTensor(class_weights).to(device)
    # print(class_weights)

    # # Get sampler
    # print("Getting sampler...")
    # sampler = WeightedRandomSampler(
    #     weights=class_weights,
    #     num_samples=len(class_weights),
    #     replacement=True
    # )

    # Get loaders
    print("Getting loaders...")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) # sampler=sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # Get model
    print("Getting model...")
    model = GCN(
        feature_size=train_set[0].x.shape[1], # 300 or 768
        model_params={
            "model_layers": 3,
            "model_dense_neurons": 32,
            "model_embedding_size": 32,
            "model_num_classes": 3,
        }
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    hist = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': [],
        'valid_x': [],
    }

    for epoch in range(0, 100):
        step_loss = 0
        step_acc = 0
        step_f1 = 0
        
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch:3d} | Train'):
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()

            pred = out.argmax(dim=1)
            acc = accuracy_score(batch.y.cpu(), pred.cpu())
            f1 = f1_score(batch.y.cpu(), pred.cpu(), average='weighted')

            step_loss += loss.item()
            step_acc += acc
            step_f1 += f1

        step_loss /= len(train_loader)
        step_acc /= len(train_loader)
        step_f1 /= len(train_loader)

        hist['train_loss'].append(step_loss)
        hist['train_acc'].append(step_acc)
        hist['train_f1'].append(step_f1)


        if epoch % 5 == 0:
            step_loss = 0
            step_acc = 0
            step_f1 = 0

            model.eval()
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc=f'Epoch {epoch:3d} | Valid'):
                    batch = batch.to(device)
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch
                    )
                    loss = criterion(out, batch.y)

                    pred = out.argmax(dim=1)
                    acc = accuracy_score(batch.y.cpu(), pred.cpu())
                    f1 = f1_score(batch.y.cpu(), pred.cpu(), average='weighted')

                    step_loss += loss.item()
                    step_acc += acc
                    step_f1 += f1

            step_loss /= len(valid_loader)
            step_acc /= len(valid_loader)
            step_f1 /= len(valid_loader)

            hist['valid_loss'].append(step_loss)
            hist['valid_acc'].append(step_acc)
            hist['valid_f1'].append(step_f1)
            hist['valid_x'].append(epoch)

        # plot the training loss and accuracy
        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        axs[0].plot(hist['train_loss'], label='train loss')
        axs[0].plot(hist['valid_x'], hist['valid_loss'], label='valid loss')
        axs[0].set_xlabel('Epoch')
        axs[0].set_ylabel('Loss')
        axs[0].legend(loc='upper right')
        axs[0].set_title('Loss')
        
        axs[1].plot(hist['train_acc'], label='train acc')
        axs[1].plot(hist['valid_x'], hist['valid_acc'], label='valid acc')
        axs[1].set_xlabel('Epoch')
        axs[1].set_ylabel('Accuracy')
        axs[1].legend(loc='upper right')
        axs[1].set_title('Accuracy')

        axs[2].plot(hist['train_f1'], label='train f1')
        axs[2].plot(hist['valid_x'], hist['valid_f1'], label='valid f1')
        axs[2].set_xlabel('Epoch')
        axs[2].set_ylabel('F1')
        axs[2].legend(loc='upper right')
        axs[2].set_title('F1')

        fig.savefig('results.png')

    print('Done!')