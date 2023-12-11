import pandas as pd
import numpy as np
import os

from src.data.utils import read_tsv
from src.data.text_preprocessing import clean_token, preprocess_text
from src.features.dataset import MyOwnDataset


if __name__ == '__main__':
    # Make directories
    os.makedirs('data/bronze', exist_ok=True)
    os.makedirs('data/silver', exist_ok=True)
    os.makedirs('data/gold', exist_ok=True)

    # Make dataset

    # MyOwnDataset('train', 'w2v', 'window', pre_transform=preprocess_text)
    # MyOwnDataset('valid', 'w2v', 'window', pre_transform=preprocess_text)
    # MyOwnDataset('test', 'w2v', 'window', pre_transform=preprocess_text)

    MyOwnDataset('train', 'w2v', 'dependency', 
                 pre_transform=preprocess_text, transform=clean_token)
    MyOwnDataset('valid', 'w2v', 'dependency',
                pre_transform=preprocess_text, transform=clean_token)
    MyOwnDataset('test', 'w2v', 'dependency',
                pre_transform=preprocess_text, transform=clean_token)