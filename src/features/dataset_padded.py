import os
import os.path as osp
import pickle as pkl
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from src.data.utils import read_tsv

class DepressionPaddedSequenceDataset(Dataset):
    def __init__(self, set_type, encoder_type, root_path="."):
        """
        Args:
            set_type (str): 'train', 'valid', or 'test'
            encoder_type (str): 'bert' or 'w2v'
        """
        root = osp.join(root_path, 'data', 'gold', set_type)

        self.root_path = root_path
        self.set_type = set_type
        self.encoder_type = encoder_type

        assert set_type in ['train', 'valid', 'test', 'development']
        assert encoder_type in ['bert', 'w2v']

        super().__init__(root)

    @property
    def raw_file_names(self):
        n_dirs = read_tsv(f'{self.root_path}/data/silver/{self.set_type}.tsv').shape[0]
        dirs = [str(i) for i in range(n_dirs)]
        return dirs

    @property
    def processed_file_names(self):
        dirs = os.listdir(self.raw_dir)
        dirs = [d for d in dirs if d.isnumeric()]
        return [f'{self.encoder_type}_padded_{d}.pt' for d in dirs]

    def download(self):
        message = f"Run `python data.py` to create the dataset."
        raise NotImplementedError(message)
        
    def process(self):
        dirs = os.listdir(self.raw_dir)
        dirs = [osp.join(self.raw_dir, d) for d in dirs if d.isnumeric()]
        
        # Get max length
        max_length = 0
        for dir in dirs:
            node_features_path = osp.join(dir, f'features_{self.encoder_type}.npy')
            node_features = np.load(node_features_path)
            max_length = max(max_length, node_features.shape[0])

        dirs = tqdm(dirs, desc=f'Processing {self.set_type} dataset')
        for i, dir in enumerate(dirs):
            # Load data
            node_features_path = osp.join(dir, f'features_{self.encoder_type}.npy')
            node_features = np.load(node_features_path)
            node_features = torch.tensor(node_features, dtype=torch.float)

            label_path = osp.join(dir, 'label.pkl')
            label = pkl.load(open(label_path, 'rb'))
            label = torch.tensor(np.asarray([label]), dtype=torch.int64)

            # Pad sequences
            padded_sequence = torch.zeros((max_length, node_features.shape[1]))
            padded_sequence[:node_features.shape[0], :] = node_features

            # Save data
            data_path = osp.join(
                self.processed_dir, 
                f'{self.encoder_type}_padded_{i}.pt'
            )

            data = Data(
                x=padded_sequence,
                y=label,
            )

            torch.save(data, data_path)

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data

    def get_weights(self):
        """
            Weights for weighted random sampling
        """
        labels = []
        for i in tqdm(range(len(self)), desc=f'Getting weights for {self.set_type} dataset'):
            data = self.get(i)
            labels.append(data.y.item())
        labels = np.asarray(labels)
        weights = torch.tensor(np.bincount(labels) / len(labels), dtype=torch.float)
        return weights