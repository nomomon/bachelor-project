import torch
import torch.nn.functional as F
from torch_geometric.nn import Linear, GATConv
from torch_geometric.nn import (
    global_mean_pool as gap, 
    global_max_pool as gmp
)

def get_edge_index(n_nodes, window_radius):
    edge_index_s = torch.cat([
        torch.arange(0, n_nodes-i).view(1, -1)
        for i in range(1, window_radius+1)
    ], dim=1)

    edge_index_e = torch.cat([
        torch.arange(i, n_nodes).view(1, -1)
        for i in range(1, window_radius+1)
    ], dim=1)

    edge_index = torch.cat([edge_index_s, edge_index_e], dim=0)

    return edge_index


class MLGCN(torch.nn.Module):
    def __init__(self, feature_size, model_params):
        super().__init__()
        self.n_layers = model_params["model_layers"]
        dense_neurons = model_params["model_dense_neurons"]
        embedding_size = model_params["model_embedding_size"]
        num_classes = model_params["model_num_classes"]
        p = model_params["model_p"]
        q = model_params["model_q"]
        # dropout = model_params["model_dropout"]
        
        self.atten_layers = torch.nn.ModuleList()
        self.linear_layers = torch.nn.ModuleList()
        # self.dropout_layer = torch.nn.Dropout(dropout)
        for i in range(self.n_layers):
            input_size = feature_size if i == 0 else embedding_size
            self.atten_layers.append(GATConv(input_size, embedding_size, heads=4))
            self.linear_layers.append(Linear(embedding_size * 4, embedding_size))

        self.line_1 = Linear(embedding_size * 2, dense_neurons)
        self.line_2 = Linear(dense_neurons, dense_neurons // 2)
        self.line_3 = Linear(dense_neurons // 2, num_classes)

    def forward(self, x, batch_index):
        global_representation = []
        
        # sliding windows with window size self.q, self.p, and fully connected
        edge_indecies = [
            get_edge_index(x.shape[0], i) for 
        ]

        for i in range(self.n_layers):
            x = self.atten_layers[i](x, edge_indecies[i])
            x = F.relu(x)
            x = self.linear_layers[i](x)
            x = F.relu(x)

            global_representation.append(torch.cat([
                gmp(x, batch_index), 
                gap(x, batch_index)
            ], dim=1))

        x = sum(global_representation)

        x = F.relu(self.line_1(x))
        # x = self.dropout_layer(x)
        x = F.relu(self.line_2(x))
        # x = self.dropout_layer(x)
        x = self.line_3(x)
        
        return x