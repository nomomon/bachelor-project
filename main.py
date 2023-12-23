from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch.optim.lr_scheduler import ExponentialLR
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits
from torch_geometric.loader import DataLoader, ImbalancedSampler

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix
)
from src.data.utils import read_tsv

from src.features.dataset import DepressionDataset
from src.models.GAT import GAT


batch_size = 64
reset_dataset = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    train_set = DepressionDataset('train', 'w2v', 'dependency')
    valid_set = DepressionDataset('valid', 'w2v', 'dependency')

    if reset_dataset:
        train_set.process()
        valid_set.process()

    # Get sampler
    print("Getting sampler...")
    sampler = ImbalancedSampler(train_set, num_samples=batch_size * 150)# len(train_set))

    # Get loaders
    print("Getting loaders...")
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    # Get model
    print("Getting model...")
    model = GAT(
        feature_size=train_set[0].x.shape[1], # 300 or 768
        model_params={
            "model_layers": 3,
            "model_dense_neurons": 64,
            "model_embedding_size": 128,
            "model_num_classes": 3 - 1,
            "model_dropout": 0.5
        }
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-2)
    scheduler = ExponentialLR(optimizer, gamma=0.9)
                                 
    hist = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'train_cm': [],
        'valid_loss': [],
        'valid_acc': [],
        'valid_f1': [],
        'valid_cm': [],
        'valid_x': [],
    }

    # Start training
    print("Start training...")
    for epoch in range(0, 50):
        step_loss = 0
        step_acc = 0
        step_f1 = 0
        step_cm = None
        
        model.train()
        for batch in tqdm(train_loader, desc=f'Epoch {epoch:3d} ┬ Train'):
            batch = batch.to(device)
            optimizer.zero_grad()

            out = model(
                batch.x,
                batch.edge_index,
                batch.batch
            )
            loss = corn_loss(out, batch.y, num_classes=3)
            loss.backward()
            optimizer.step()

            pred = corn_label_from_logits(out)
            acc = accuracy_score(batch.y.cpu(), pred.cpu())
            f1 = f1_score(batch.y.cpu(), pred.cpu(), average='macro')
            # roc_auc = roc_auc_score(batch.y.cpu(), pred.cpu(), average='weighted')

            step_loss += loss.item()
            step_acc += acc
            step_f1 += f1
            batch_cm = confusion_matrix(batch.y.cpu(), pred.cpu(), labels=[0, 1, 2])
            step_cm = batch_cm if step_cm is None else step_cm + batch_cm

        step_loss /= len(train_loader)
        step_acc /= len(train_loader)
        step_f1 /= len(train_loader)

        hist['train_loss'].append(step_loss)
        hist['train_acc'].append(step_acc)
        hist['train_f1'].append(step_f1)
        hist['train_cm'].append(step_cm)

        scheduler.step()

        if epoch % 1 == 0:
            step_loss = 0
            step_acc = 0
            step_f1 = 0
            step_cm = None

            model.eval()
            with torch.no_grad():
                for batch in tqdm(valid_loader, desc='          └ Valid'):
                    batch = batch.to(device)
                    out = model(
                        batch.x,
                        batch.edge_index,
                        batch.batch
                    )
                    loss = corn_loss(out, batch.y, num_classes=3)

                    pred = corn_label_from_logits(out)
                    acc = accuracy_score(batch.y.cpu(), pred.cpu())
                    f1 = f1_score(batch.y.cpu(), pred.cpu(), average='macro')

                    step_loss += loss.item()
                    step_acc += acc
                    step_f1 += f1
                    batch_cm = confusion_matrix(batch.y.cpu(), pred.cpu(), labels=[0, 1, 2])
                    step_cm = batch_cm if step_cm is None else step_cm + batch_cm

            step_loss /= len(valid_loader)
            step_acc /= len(valid_loader)
            step_f1 /= len(valid_loader)

            hist['valid_loss'].append(step_loss)
            hist['valid_acc'].append(step_acc)
            hist['valid_f1'].append(step_f1)
            hist['valid_cm'].append(step_cm)
            hist['valid_x'].append(epoch)

        # plot the training loss and accuracy
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs[0][0].plot(hist['train_loss'], label='train loss')
        axs[0][0].plot(hist['valid_x'], hist['valid_loss'], label='valid loss')
        axs[0][0].set_xlabel('Epoch')
        axs[0][0].set_ylabel('Loss')
        axs[0][0].legend(loc='upper right')
        axs[0][0].set_title('Loss')
        
        axs[0][1].plot(hist['train_acc'], label='train acc')
        axs[0][1].plot(hist['valid_x'], hist['valid_acc'], label='valid acc')
        axs[0][1].set_xlabel('Epoch')
        axs[0][1].set_ylabel('Accuracy')
        axs[0][1].legend(loc='upper right')
        axs[0][1].set_title('Accuracy')

        axs[0][2].plot(hist['train_f1'], label='train f1')
        axs[0][2].plot(hist['valid_x'], hist['valid_f1'], label='valid f1')
        axs[0][2].set_xlabel('Epoch')
        axs[0][2].set_ylabel('F1')
        axs[0][2].legend(loc='upper right')
        axs[0][2].set_title('F1 (macro)')

        sns.heatmap(hist['train_cm'][-1], annot=True, fmt='d', ax=axs[1][0])
        axs[1][0].set_xlabel('Predicted')
        axs[1][0].set_ylabel('Actual')
        axs[1][0].set_title(f'Train Confusion Matrix (Epoch {epoch:3d})')
        axs[1][0].set_aspect('equal')

        sns.heatmap(hist['valid_cm'][-1], annot=True, fmt='d', ax=axs[1][1])
        axs[1][1].set_xlabel('Predicted')
        axs[1][1].set_ylabel('Actual')
        axs[1][1].set_title(f'Valid Confusion Matrix (Epoch {hist["valid_x"][-1]:3d})')
        axs[1][1].set_aspect('equal')

        sns.heatmap(hist['valid_cm'][-1] / hist['valid_cm'][-1].sum(axis=1)[:, np.newaxis], 
                    annot=True, fmt='.2%', ax=axs[1][2])
        axs[1][2].set_xlabel('Predicted')
        axs[1][2].set_ylabel('Actual')
        axs[1][2].set_title(f'Valid Confusion Matrix (Normed)')
        axs[1][2].set_aspect('equal')

        fig.savefig('results.png')
        plt.close()

    print('Done!')