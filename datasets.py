import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from src.data.utils import read_tsv
from src.data.text_preprocessing import preprocess_text
from src.features.embeddings import get_node_features_bert, get_node_features_w2v
from src.features.graphs import pipe_get_graph_dependency


# splits = ['development']
splits = ['train', 'valid', 'test']

if __name__ == '__main__':
    assert os.path.exists('data/bronze/')

    # Make directories
    os.makedirs('data/silver', exist_ok=True)
    os.makedirs('data/gold', exist_ok=True)


    # Clean data
    print('\nCleaning data...')
    for split in splits:
        df = read_tsv(f'data/bronze/{split}.tsv')
        df['text'] = df['text'].apply(preprocess_text)
        df.to_csv(f'data/silver/{split}.tsv', sep='\t', index=False)
    

    # Create graphs
    print('\nCreating graphs...')
    for split in splits:
        os.makedirs(f'data/gold/{split}/raw', exist_ok=True)
        os.makedirs(f'data/gold/{split}/processed', exist_ok=True)

        df = pd.read_csv(f'data/silver/{split}.tsv', sep='\t')

        # Get dependency graphs
        texts = df['text'] 
        texts = tqdm(texts, desc=f'{split:5} | Dependency graphs')
        all_nodes, all_edges = pipe_get_graph_dependency(texts)

        # Save graphs
        with open(f'data/gold/{split}/raw/all_nodes.pkl', 'wb') as f:
            pickle.dump(all_nodes, f)
        with open(f'data/gold/{split}/raw/all_edges.pkl', 'wb') as f:
            pickle.dump(all_edges, f)


    # Get split into subdirectories
    print('\nSplitting into subdirectories...')
    for split in splits:
        all_nodes = pickle.load(open(f'data/gold/{split}/raw/all_nodes.pkl', 'rb'))
        all_edges = pickle.load(open(f'data/gold/{split}/raw/all_edges.pkl', 'rb'))

        # Split into subdirectories
        for i in tqdm(range(len(all_nodes)), desc=f'{split:5} | Subdirectories'):
            os.makedirs(f'data/gold/{split}/raw/{i}', exist_ok=True)
            with open(f'data/gold/{split}/raw/{i}/nodes.pkl', 'wb') as f:
                pickle.dump(all_nodes[i], f)
            with open(f'data/gold/{split}/raw/{i}/edges.pkl', 'wb') as f:
                pickle.dump(all_edges[i], f)


    # Get Word2Vec embeddings
    print('\nGetting Word2Vec embeddings...')
    for split in splits:
        dirs = os.listdir(f'data/gold/{split}/raw/')
        dirs = [d for d in dirs if d.isnumeric()]
        dirs = tqdm(dirs, desc=f'{split:5} | Word2Vec')

        for d in dirs:
            nodes = pickle.load(open(f'data/gold/{split}/raw/{d}/nodes.pkl', 'rb'))
            features_w2v = get_node_features_w2v(nodes)
            np.save(f'data/gold/{split}/raw/{d}/features_w2v.npy', features_w2v)


    # Get BERT embeddings
    print('\nGetting BERT embeddings...')    
    for split in splits:
        dirs = os.listdir(f'data/gold/{split}/raw/')
        dirs = [d for d in dirs if d.isnumeric()]
        dirs = tqdm(dirs, desc=f'{split:5} | BERT')

        for d in dirs:
            nodes = pickle.load(open(f'data/gold/{split}/raw/{d}/nodes.pkl', 'rb'))
            features_bert = get_node_features_bert(nodes)
            np.save(f'data/gold/{split}/raw/{d}/features_bert.npy', features_bert)
