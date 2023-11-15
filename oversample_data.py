#%%
import pandas as pd

data = pd.read_csv("data/raw/train.tsv", sep="\t", index_col=0)

not_depression_class  = data[data["Label"] == "not depression"]
moderate_class       = data[data["Label"] == "moderate"]
severe_class         = data[data["Label"] == "severe"]

#%%
# show sizes of each class
print("not_depression_class:", not_depression_class.shape)
print("moderate_class:", moderate_class.shape)
print("severe_class:", severe_class.shape)
# %%

ratio_1_0 = int(moderate_class.shape[0]/not_depression_class.shape[0]) - 1
ratio_1_2 = int(moderate_class.shape[0]/severe_class.shape[0]) - 1

replicated_1_0 = [not_depression_class]*ratio_1_0
replicated_1_2 = [severe_class]*ratio_1_2

data = pd.concat([data, *replicated_1_0, *replicated_1_2], ignore_index=True)

#%%

# shuffle dataset
data = data.sample(frac=1).reset_index(drop=True)

# re-assign index
data.index = range(0, data.shape[0])
data.index = data.index.map(lambda x: "train_oversampled_pid_" + str(x))
data.index.name = "PID"

#%%
# save

data.to_csv("data/raw/train_oversampled.tsv", sep="\t")
# %%
