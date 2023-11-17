import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap, 
    global_max_pool as gmp
)

class GCN(torch.nn.Module):
    def __init__(self, feature_size, num_classes, model_params):
        super().__init__()
        embedding_size = model_params["model_embedding_size"]
        # n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        # dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        dense_neurons = model_params["model_dense_neurons"]

        self.conv_1 = GCNConv(feature_size, embedding_size)
        self.conv_2 = GCNConv(embedding_size, embedding_size)
        self.conv_3 = GCNConv(embedding_size, embedding_size)
        self.pool_1 = TopKPooling(embedding_size, ratio=top_k_ratio)
        self.pool_2 = TopKPooling(embedding_size, ratio=top_k_ratio)
        self.pool_3 = TopKPooling(embedding_size, ratio=top_k_ratio)
        self.line_1 = Linear(embedding_size*2, dense_neurons)
        self.line_2 = Linear(dense_neurons, dense_neurons // 2)
        self.line_3 = Linear(dense_neurons // 2, num_classes)

        self.conv_layers = [self.conv_1, self.conv_2, self.conv_3]
        self.pool_layers = [self.pool_1, self.pool_2, self.pool_3]
        self.line_layers = [self.line_1, self.line_2, self.line_3]

    def forward(self, x, edge_index, batch_index):
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)
            x, edge_index, _, batch_index, _, _ = self.pool_layers[i](
                x, edge_index, None, batch_index
            )
            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))

        x = sum(global_representation)

        x = F.relu(self.line_1(x))
        x = F.relu(self.line_2(x))
        x = self.line_3(x)

        return F.log_softmax(x, dim=1)