import numpy as np
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, TensorSpec

# TODO: clean up the unnecessary hyperparameters
HYPERPARAMETERS = {
    "batch_size": [32, 128, 64],
    "learning_rate": [0.1, 0.05, 0.01, 0.001],
    "weight_decay": [0.0001, 0.00001, 0.001],
    "sgd_momentum": [0.9, 0.8, 0.5],
    "scheduler_gamma": [0.995, 0.9, 0.8, 0.5, 1],
    "pos_weight" : [1.0],  
    "model_embedding_size": [8, 16, 32, 64, 128],
    "model_attention_heads": [1, 2, 3, 4],
    "model_layers": [3],
    "model_dropout_rate": [0.2, 0.5, 0.9],
    "model_top_k_ratio": [0.2, 0.5, 0.8, 0.9],
    "model_top_k_every_n": [0],
    "model_dense_neurons": [16, 128, 64, 256, 32]
}

# TODO: save the best hyperparameters here

# Input schema
# TODO: update the schema to match your features
# TODO: write the explanation of the schema
input_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 30), name="x"), 
                       TensorSpec(np.dtype(np.int32), (2, -1), name="edge_index"), 
                       TensorSpec(np.dtype(np.int32), (-1, 1), name="batch_index")])

# Output schema
# a list of tensors, where each is a vector of 3 probabilities
output_schema = Schema([TensorSpec(np.dtype(np.float32), (-1, 3))])

SIGNATURE = ModelSignature(inputs=input_schema, outputs=output_schema)