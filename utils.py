import numpy as np
import sklearn.utils.class_weight as scikit_class_weight

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def compute_class_weight(train_y):
    """
    Compute class weight given imbalanced training data
    Usually used in the neural network model to augment the loss function (weighted loss function)
    Favouring/giving more weights to the rare classes.
    """

    class_list = sorted(list(set(train_y)))
    class_weight_value = scikit_class_weight.compute_class_weight(
        class_weight ='balanced', 
        classes = class_list, 
        y = train_y
    )
    return class_weight_value.tolist()


import os
os.system('cls' if os.name == 'nt' else 'clear')
print(bcolors.HEADER, "Running utils.py, cleared terminal", bcolors.ENDC)