import spacy
import torch
import numpy as np

nlp = spacy.load("en_core_web_lg")

def pipe_get_graph_dependency(texts):
    docs = list(nlp.pipe(texts))
    all_nodes = [[token.text for token in doc] for doc in docs]
    all_edges = []

    for doc in docs:
        edges = []
        for token in doc:
            for child in token.children:
                edges.append([token.i, child.i])
                edges.append([child.i, token.i])
        all_edges.append(edges)
        
    assert len(all_nodes) == len(all_edges)

    return all_nodes, all_edges