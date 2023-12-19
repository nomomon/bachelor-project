import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, Linear, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap, 
    global_max_pool as gmp
)

class GAT(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_dense_neurons"]
        embedding_size = model_params["model_embedding_size"]
        num_classes = model_params["model_num_classes"]
        dropout = model_params["model_dropout"]
        
        self.atten_layers = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_layers):
            input_size = feature_size if i == 0 else embedding_size
            self.atten_layers.append(GATConv(input_size, embedding_size, heads=4, dropout=dropout))
            self.linear_layers.append(Linear(embedding_size * 4, embedding_size))

        self.line_1 = Linear(embedding_size * 2, dense_neurons)
        self.line_2 = Linear(dense_neurons, dense_neurons // 2)
        self.line_3 = Linear(dense_neurons // 2, num_classes)

    def forward(self, x, edge_index, batch_index):
        global_representation = []

        for i in range(self.n_layers):
            x = self.atten_layers[i](x, edge_index)
            x = F.relu(x)
            
            x = self.linear_layers[i](x)
            x = F.relu(x)

            # Skip connection
            if i > 0:
                x = x + previous_x
            previous_x = x

        x = torch.cat([
            gmp(x, batch_index), 
            gap(x, batch_index)
        ], dim=1)

        x = F.relu(self.line_1(x))
        x = F.relu(self.line_2(x))
        x = self.line_3(x)
        
        return x