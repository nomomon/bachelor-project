ask Habrock access for bach proj (cc to prof)

make a model architecture diagram for how it works

look at the gits of papers
- change from of their code (not copy paste what they did)

compare and explain the results, give the whys on something happened

just use tfidf, bert, w2v

send a progress report with points of discussion

https://medium.com/saarthi-ai/sentence-classification-using-convolutional-neural-networks-ddad72c7048c

https://theaisummer.com/attention/

https://github.com/GeneZC/ASGCN

https://github.com/Daisean/HINT/blob/main/models/coding_tree_learning.py

# Detecting Signs of Depression from Social Media Text with GNNs

In this project, we use a graph neural network (GNN) to predict depression from text. Text can be modeled as graphs in various ways: assuming words are nodes they can be connected by edges if they are adjacent in a sentence, or the edges can be relations from sentence parsing, or similarly to how humans read – in chunks of text like n-grams.

The dataset used in <--CITE--> which contains posts from Reddit. The dataset is split into training, validation, and test sets.

| Set | # of posts | # of not | # of moderate | # of severe |
| --- | --- | --- | --- | --- |
| Train | 8891 | 1971 | 6019 | 901 |
| Dev | 4496 | 1830 | 2306 | 360 |
| Test | 3245 | 848 | 2169 | 228 |


## Overview of Files and Scripts

Scripts:

- `oversample_data.py` – the classes in dataset are imbalanced, the script oversamples the training set to balance the classes and creates a new file `train_oversampled.csv` with the new data.

- `dataset.py` – implements a PyTorch Dataset class for the dataset, which is later used in the DataLoader. Currently, it implements a simple graph of adjecent words in a sentence with Word2Vec embeddings for nodes.

- `model.py` – implements a GNN model with a Graph Attention Convolution layer and a linear layer for classification.

- `train.py` – trains the model on the training set and evaluates on the validation set. The model parameters and results are logged into mlflow.

- `utils.py` – contains helper functions: color logging and train weight balancer.

- `config.py` – (TODO) finds stores and implemets the search for the best hyperparameters.

Directories:

- `mlruns` – contains the mlflow logs for the runs.

- `data` – contains the dataset files.