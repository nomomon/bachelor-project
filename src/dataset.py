import numpy as np
import re
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

import os

class DepressionDataset(Dataset):
    """
    Depression dataset class for PyTorch Geometric

    Based on the *DOI* of the dataset:
    """

    label_to_class = {"not depression": 0, "moderate": 1, "severe": 2}

    def __init__(self, 
        root, filename, 
        prefix="train", 
        transform=None, 
        pre_transform=None, 
        word_encoder=None,
        graph_encoder=None,
        raw_data=None
    ):
        """
        root = where the dataset should be stored. This folder is split
        into raw_dir (downloaded dataset) and processed_dir (processed data).
        """

        self.prefix = prefix
        self.filename = filename

        self.word_encoder = word_encoder
        self.graph_encoder = graph_encoder

        self.raw_data = raw_data

        super(DepressionDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """If this file exists in raw_dir, the download is not triggered.
        (The download func. is not implemented here)
        """
        return self.filename

    @property
    def processed_file_names(self):
        """If these files are found in raw_dir, processing is skipped"""
        return [f"data_{self.prefix}_{i}.pt" for i in list(self.raw_data.index)]

    def download(self):
        """
        Download the dataset to raw_dir
        """
        # TODO: Implement this later
        pass

    def process(self):
        if self.word_encoder == None or self.graph_encoder == None:
            raise Exception("word_encoder or graph_encoder are not implemented")

        for index, row in tqdm(self.raw_data.iterrows(), total=self.raw_data.shape[0]):
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
                edge_index=edge_index, 
                pid=pid
            )
            
            torch.save(data, os.path.join(self.processed_dir, f"data_{self.prefix}_{index}.pt"))

    def decontracted(self, text):
        # specific
        text = re.sub(r"won\'t", "will not", text)
        text = re.sub(r"can\'t", "can not", text)
        text = re.sub(r"gotta", "got to", text)

        # general
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"\'s", " is", text)
        text = re.sub(r"\'d", " would", text)
        text = re.sub(r"\'ll", " will", text)
        text = re.sub(r"\'t", " not", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'m", " am", text)
        return text

    def _preprocess_text(self, text):
        text = text.lower()
        text = self.decontracted(text)
        text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
        text = re.sub(r"\s+", " ", text)
        text = text.split()

        return text

    def _get_node_features(self, word_list):
        """
        This will return a matrix / 2d array of the shape
        [Number of Nodes, Node Feature size]
        """
        all_node_feats = self.word_encoder(word_list)
        return torch.tensor(all_node_feats, dtype=torch.float)

    def _get_edge_features(self, word_list):
        """
        There are no edge features in this dataset
        """
        pass

    def _get_adjacency_info(self, word_list):
        """
        Simple adjacency info for now. Words are connected if they are
        next to each other in the sentence.
        """
        edge_indices = self.graph_encoder(word_list)

        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        return edge_indices

    def _get_labels(self, label):
        label = self.label_to_class[label]
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    def get_targets(self):
        """
        Returns the tragets of the dataset
        """
        labels = self.raw_data["label"].values
        targets = np.array([self.label_to_class[label] for label in labels])
        return targets

    def len(self):
        """
        Equivalent to __len__ in pytorch
        Is not needed for PyG's InMemoryDataset
        """
        return len(self.raw_data)

    def get(self, idx):
        """
        Equivalent to __getitem__ in pytorch
        Is not needed for PyG's InMemoryDataset
        """

        filename = f'data_{self.prefix}_{idx}.pt'

        data = torch.load(os.path.join(self.processed_dir, filename))
        return data