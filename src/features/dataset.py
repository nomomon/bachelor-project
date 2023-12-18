import os
import shutil
import os.path as osp

import numpy as np
from tqdm import tqdm
import spacy
import gensim.downloader as gensim_api
from transformers import BertTokenizer, BertModel

import torch
from torch_geometric.data import Dataset, Data

from src.data.utils import read_tsv


class MyOwnDataset(Dataset):
    def __init__(self, set_type, encoder_type, graph_type,
                transform=None, pre_transform=None, pre_filter=None):
        """
        Args:
            set_type (str): 'train', 'valid', or 'test'
            encoder_type (str): 'bert' or 'w2v'
            graph_type (str): 'window' or 'dependency'
            pre_transform (callable, optional): Text preprocessing function.
            transform (callable, optional): Word transformation function.
        """
        
        dir_name = f"{set_type}_{encoder_type}_{graph_type}"
        root = osp.join('data', 'gold', dir_name)

        self.set_type = set_type
        self.encoder_type = encoder_type
        self.graph_type = graph_type
        
        super().__init__(root, transform, pre_transform, pre_filter)



    @property
    def raw_file_names(self):
        return [f'{self.set_type}.tsv']

    @property
    def processed_file_names(self):
        df = read_tsv(self.raw_paths[0])
        return [f'data_{idx}.pt' for idx in df.index]

    def download(self):
        # Copy `data/bronze/{set_type}.tsv` to `self.raw_dir`.
        os.makedirs(self.raw_dir, exist_ok=True)
        path = osp.join('data', 'bronze', f'{self.set_type}.tsv')
        shutil.copy(path, self.raw_dir)
        
    def process(self):
        df = read_tsv(self.raw_paths[0])
        rows = tqdm(df.iterrows(), total=len(df), desc=f'{self.raw_paths[0]}')

        for idx, row in rows:
            text = row.text
            label = row.label

            if self.pre_transform is not None:
                text = self.pre_transform(text)

            nodes, edges = self._get_graph(text)

            if self.transform is not None:
                nodes = [self.transform(node) for node in nodes]

            node_features = self._get_node_features(nodes)
            
            data = Data(
                x=node_features,
                edge_index=edges,
                y=self._label_to_tensor(label)
            )

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        filename = f'data_{self.set_type}_pid_{idx + 1}.pt'
        data = torch.load(osp.join(self.processed_dir, filename))
        return data
    
    def _label_to_tensor(self, label):
        label_to_class = {"not depression": 0, "moderate": 1, "severe": 2}
        label = label_to_class[label]
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.int64)
    
    def _get_graph(self, text):
        if self.graph_type == 'window':
            nodes, edges = self._get_graph_window(text)
        elif self.graph_type == 'dependency':
            nodes, edges = self._get_graph_dependency(text)
        else:
            raise ValueError(f'Invalid graph type: {self.graph_type}')
        
        edges = torch.tensor(edges)
        edges = edges.t().to(torch.long).view(2, -1)
        
        return nodes, edges
    
    def _get_graph_window(self, text, window_radius=5):
        nodes = text.split() # list of words

        edges = []
        for i in range(len(nodes)):
            for j in range(i - window_radius, i + window_radius + 1):
                if i != j and 0 <= j < len(nodes):
                    edges.append([i, j])
        edges = np.array(edges)

        return nodes, edges
    
    
    def _get_graph_dependency(self, text):
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)

        nodes = [token.text for token in doc]
        edges = []
        for token in doc:
            for child in token.children:
                edges.append([token.i, child.i])
                edges.append([child.i, token.i])
        edges = np.array(edges)

        return nodes, edges
        
    
    def _get_node_features(self, nodes):
        if self.encoder_type == 'bert':
            node_features = self._get_node_features_bert(nodes)
        elif self.encoder_type == 'w2v':
            node_features = self._get_node_features_w2v(nodes)
        else:
            raise ValueError(f'Invalid encoder type: {self.encoder_type}')
        
        return node_features
    
    
    def _get_node_features_w2v(self, nodes):
        if hasattr(self, 'w2v_model') is False:
            self.w2v_model = gensim_api.load("word2vec-google-news-300")

        node_features = []
        for node in nodes:
            try:
                node_features.append(self.w2v_model[node])
            except KeyError:
                node_features.append(np.random.normal(size=300, scale=0.12))
        node_features = np.stack(node_features, axis=0)
        node_features = torch.tensor(node_features, dtype=torch.float)

        return node_features
    
    def _get_node_features_bert(self, nodes):
        if hasattr(self, 'bert_tokenizer') and hasattr(self, 'bert_model') is False:
            self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()

        node_features = []
        for node in nodes:
            input_ids = torch.tensor([self.bert_tokenizer.encode(node)])
            with torch.no_grad():
                last_hidden_states = self.bert_model(input_ids)[0]
            node_features.append(last_hidden_states.squeeze(0).mean(dim=0))
        node_features = torch.stack(node_features, dim=0)

        return node_features
    
    