# %%
import pandas as pd

import torch
import torch_geometric
from torch_geometric.data import Dataset, Data

import numpy as np 
import os
from tqdm import tqdm

# %%
print(f"Torch version: {torch.__version__}")
print(f"Cuda available: {torch.cuda.is_available()}")
print(f"Torch geometric version: {torch_geometric.__version__}")

# %%
from gensim.models import Word2Vec
import gensim.downloader as api

w2v_model = api.load('word2vec-google-news-300')
vocab = w2v_model.key_to_index

# %%
import os.path as osp
import re

class DepressionDataset(Dataset):
    def __init__(self, root, filename, test=False, transform=None, pre_transform=None):
        """
        root = Where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data). 
        """
        self.test = test
        self.filename = filename

        self.label_to_class = {
            "not depression": 0,
            "moderate": np.random.randint(0, 2),
            "severe": 1,
        }

        super(DepressionDataset, self).__init__(root, transform, pre_transform)
        
    @property
    def raw_file_names(self):
        """ If this file exists in raw_dir, the download is not triggered.
            (The download func. is not implemented here)  
        """
        return self.filename

    @property
    def processed_file_names(self):
        """ If these files are found in raw_dir, processing is skipped"""
        self.data = pd.read_csv(self.raw_paths[0], sep="\t", index_col="PID").reset_index()


        if self.test:
            return [f'data_test_{i}.pt' for i in list(self.data.index)]
        else:
            return [f'data_{i}.pt' for i in list(self.data.index)]

    def download(self):
        pass

    def process(self):
        self.data = pd.read_csv(self.raw_paths[0], sep="\t", index_col="PID").reset_index()
        
        for index, row in tqdm(self.data.iterrows(), total=self.data.shape[0]):
            pid, text, class_label = row.values
            word_list = self._preprocess_text(text)
            

            # Get node features
            node_feats = self._get_node_features(word_list)
            # Get adjacency info
            edge_index = self._get_adjacency_info(word_list)
            # Get labels info
            label = self._get_labels(class_label)

            # Create data object
            data = Data(
                x=node_feats, y=label,
                edge_index=edge_index, pid=pid
            )

            if self.test:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_test_{index}.pt'))
            else:
                torch.save(data, 
                    os.path.join(self.processed_dir, 
                                 f'data_{index}.pt'))

    def _preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"[^a-zA-Z0-9]+", ' ', text)
        text = re.sub(r"\s+", ' ', text)
        text = text.split()

        return text

    def _get_node_features(self, word_list):
        """ 
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = []

        for word in word_list:
            if word in vocab:
                node_feats = w2v_model[word]
            else: 
                node_feats = np.zeros(w2v_model.vector_size)
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, word_list):
        # there are no edge features in this dataset
        # or not that I know of
        pass

    def _get_adjacency_info(self, word_list):
        """
        Simple adjacency info for now. Words are connected if they are
        next to each other in the sentence.
        """
        edge_indices = []
        for i in range(len(word_list)-1):
            edge_indices.append([i, i+1])
            edge_indices.append([i+1, i])

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = self.label_to_class[label]
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)

    def len(self):
        return self.data.shape[0]

    def get(self, idx):
        """ - Equivalent to __getitem__ in pytorch
            - Is not needed for PyG's InMemoryDataset
        """
        
        filename = f'data{"_test" if self.test else ""}_{idx}.pt'

        data = torch.load(os.path.join(self.processed_dir, filename))
        return data
# %%

# dataset = DepressionDataset(root='data', filename='train.tsv', test=False)
