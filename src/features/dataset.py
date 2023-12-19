import os
import os.path as osp

import numpy as np
import pickle as pkl
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

from src.data.utils import read_tsv


class DepressionDataset(Dataset):
    def __init__(self, set_type, encoder_type, graph_type):
        """
        Args:
            set_type (str): 'train', 'valid', or 'test'
            encoder_type (str): 'bert' or 'w2v'
            graph_type (str): 'dependency' or 'other'
        """
        
        root = osp.join('data', 'gold', set_type)

        self.set_type = set_type
        self.encoder_type = encoder_type
        self.graph_type = graph_type

        assert set_type in ['train', 'valid', 'test', 'development']
        assert encoder_type in ['bert', 'w2v']
        assert graph_type in ['dependency', 'other']
        
        super().__init__(root)

    @property
    def raw_file_names(self):
        n_dirs = read_tsv(f'data/silver/{self.set_type}.tsv').shape[0]
        dirs = [str(i) for i in range(n_dirs)]

        return dirs
        

    @property
    def processed_file_names(self):
        dirs = os.listdir(self.raw_dir)
        dirs = [d for d in dirs if d.isnumeric()]

        return [f'{self.encoder_type}_{self.graph_type}_{d}.pt' for d in dirs]

    def download(self):
        message = f"Run `python data.py` to create the dataset."
        raise NotImplementedError(message)
        
    def process(self):
        dirs = os.listdir(self.raw_dir)
        dirs = [osp.join(self.raw_dir, d) for d in dirs if d.isnumeric()]
        dirs = tqdm(dirs, desc=f'Processing {self.set_type} dataset')

        for i, dir in enumerate(dirs):
            # Load data
            node_features_path = osp.join(dir, f'features_{self.encoder_type}.npy')
            node_features = np.load(node_features_path)
            node_features = torch.tensor(node_features, dtype=torch.float)

            label_path = osp.join(dir, 'label.pkl')
            label = pkl.load(open(label_path, 'rb'))
            label = torch.tensor(np.asarray([label]), dtype=torch.int64)

            if self.graph_type == 'dependency':
                edges_path = osp.join(dir, f'edges.pkl')
                edges = pkl.load(open(edges_path, 'rb'))
                edges = torch.tensor(edges)
                edges = edges.t().to(torch.long).view(2, -1)
            else:
                edges = None

            data = Data(
                edge_index=edges,
                x=node_features,
                y=label
            )

            data_path = osp.join(
                self.processed_dir, 
                f'{self.encoder_type}_{self.graph_type}_{i}.pt'
            )
            torch.save(data, data_path)

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, self.processed_file_names[idx]))
        return data