import torch
import os
import numpy as np
import pickle
from torch.utils.data import Dataset

class DepressionDataset(Dataset):
    def __init__(self, set_type, encoder_type, root_path="."):
        
        self.set_type = set_type
        self.encoder_type = encoder_type
        self.root_path = root_path

        self.raw_dir = os.path.join(root_path, "data", "gold", set_type, "raw")

        assert os.path.exists(self.raw_dir), f"Path {self.raw_dir} does not exist"

        self.sample_weights = self.get_sample_weights()

    def __len__(self):
        dirs = os.listdir(self.raw_dir)
        dirs = [d for d in dirs if os.path.isdir(os.path.join(self.raw_dir, d))]
        return len(dirs)

    def __getitem__(self, idx):
        idx = str(idx)
        features_path = os.path.join(self.raw_dir, idx, f"features_{self.encoder_type}.npy")
        label_path = os.path.join(self.raw_dir, idx, "label.pkl")

        features = np.load(features_path)
        label = pickle.load(open(label_path, "rb"))

        features = torch.from_numpy(features).float()
        label = torch.tensor(label).long()

        return features, label
    
    def get_sample_weights(self):
        labels = []
        for idx in range(len(self)):
            idx = str(idx)
            label_path = os.path.join(self.raw_dir, idx, "label.pkl")
            label = pickle.load(open(label_path, "rb"))
            labels.append(label)
        labels = np.array(labels)

        class_weights = np.zeros(np.unique(labels).shape[0])
        for i in range(len(class_weights)):
            class_weights[i] = len(labels) / np.sum(labels == i)
        
        sample_weights = class_weights[labels]

        return sample_weights
    
def collate_fn(batch):
    features, labels = zip(*batch)
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True)
    return features, torch.stack(labels)


def get_sampler(dataset: Dataset):
    return torch.utils.data.sampler.WeightedRandomSampler(
        weights=dataset.sample_weights,
        num_samples=len(dataset.sample_weights),
        replacement=True
    )