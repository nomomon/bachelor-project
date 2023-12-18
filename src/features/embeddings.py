import gensim.downloader as gensim_api
import torch
from transformers import BertTokenizer, BertModel
import numpy as np

w2v_model = gensim_api.load("word2vec-google-news-300")

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')
bert_model.eval()


def get_node_features_w2v(nodes):
    global w2v_model
    node_features = []

    for node in nodes:
        try:
            node_features.append(w2v_model[node])
        except KeyError:
            node_features.append(np.random.normal(size=300, scale=0.12))

    node_features = np.stack(node_features, axis=0)
    node_features = node_features.astype(np.float32)

    assert node_features.shape[1] == 300
    assert node_features.shape[0] == len(nodes)

    return node_features



def get_node_features_bert(nodes):
    node_features = []

    embeddings = []
    for i in range(0, len(nodes), 510):
        emb = get_bert_embeddings(nodes[i:i+510])
        embeddings.append(emb)

    if len(node_features) == 1:
        node_features = embeddings[0]
    else:
        node_features = torch.cat(embeddings, dim=0)
    node_features = node_features.numpy()

    assert node_features.shape[0] == len(nodes)
    assert node_features.shape[1] == 768

    return node_features

def get_bert_embeddings(nodes):
    assert len(nodes) <= 510
    tokens = ['[CLS]', *nodes, '[SEP]']
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([token_ids])

    with torch.no_grad():
        last_hidden_state = bert_model(input_tensor)[0][0]

    return last_hidden_state[1:-1]

