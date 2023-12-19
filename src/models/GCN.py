import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, Linear, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap, 
    global_max_pool as gmp
)

class GCN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_dense_neurons"]
        embedding_size = model_params["model_embedding_size"]
        num_classes = model_params["model_num_classes"]
        # dropout = model_params["model_dropout"]
        
        self.conv_layers = torch.nn.ModuleList()
        self.pooling_layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            input_size = feature_size if i == 0 else embedding_size
            self.conv_layers.append(GCNConv(input_size, embedding_size))
            self.pooling_layers.append(TopKPooling(embedding_size, ratio=0.2))

        self.line_1 = Linear(embedding_size * 2, dense_neurons)
        self.line_2 = Linear(dense_neurons, dense_neurons // 2)
        self.line_3 = Linear(dense_neurons // 2, num_classes)

    def forward(self, x, edge_index, batch_index):
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = F.relu(x)

            x, edge_index, _, batch_index, _, _ = self.pooling_layers[i](
                x, edge_index, None, batch_index
            )
            
            global_representation.append(torch.cat([
                gmp(x, batch_index), 
                gap(x, batch_index)
            ], dim=1))

        x = sum(global_representation)

        x = F.relu(self.line_1(x))
        x = F.relu(self.line_2(x))
        
        return x