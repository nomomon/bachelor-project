import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList
from torch_geometric.nn import GATConv, TopKPooling
from torch_geometric.nn import (
    global_mean_pool as gap, 
    global_max_pool as gmp
)

class GNN(torch.nn.Module):
    def __init__(self, feature_size, num_classes, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        n_heads = model_params["model_attention_heads"]
        self.n_layers = model_params["model_layers"]
        dropout_rate = model_params["model_dropout_rate"]
        top_k_ratio = model_params["model_top_k_ratio"]
        dense_neurons = model_params["model_dense_neurons"]

        self.conv_layers = ModuleList([])
        self.head_layers = ModuleList([])
        self.pooling_layers = ModuleList([])

        if self.n_layers == 0:
            raise NotImplementedError("Number of layers must be greater than 0")

        for i in range(self.n_layers):
            input_size = feature_size if i == 0 else embedding_size

            self.conv_layers.append(GATConv(input_size, embedding_size, heads=n_heads, dropout=dropout_rate))
            self.head_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.pooling_layers.append(TopKPooling(embedding_size, ratio=top_k_ratio))

        self.linear_1 = Linear(embedding_size * 2, dense_neurons)
        self.linear_2 = Linear(dense_neurons, dense_neurons // 2)
        self.linear_3 = Linear(dense_neurons // 2, num_classes)

    def forward(self, x, edge_index, batch_index):
        global_representation = []

        for i in range(self.n_layers):
            x = self.conv_layers[i](x, edge_index)
            x = self.head_layers[i](x)
            x, edge_index, _, batch_index, _, _ = self.pooling_layers[i](
                x, edge_index, None, batch_index
            )

            global_representation.append(torch.cat([gmp(x, batch_index), gap(x, batch_index)], dim=1))
        
        x = sum(global_representation)

        # Output block
        x = torch.relu(self.linear_1(x))
        x = F.dropout(x, p=0.8, training=self.training) # TODO: does this dropout need to be 0.8?
        x = torch.relu(self.linear_2(x))                # can we use the dropout from config?
        x = F.dropout(x, p=0.8, training=self.training)
        x = self.linear_3(x)

        return x
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)