import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import timeit
import numpy as np
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

    torch.manual_seed(1234)
    dataset = SMRTDatasetRetained('./SMRT_Retain')
    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_len_2 = int(train_len * 0.9)
    dev_len = train_len - train_len_2
    train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])
    selected, no_selected = random_split(test_dataset, [100, len(test_dataset) - 100])
    no_s = np.array([selected[0].fingerprint.numpy()])
    for i in range(len(no_selected)):
        no_s = np.append(no_s, [no_selected[i].fingerprint.numpy()], axis=0)

    yes_s = np.array([selected[0].fingerprint.numpy()])
    for i in range(len(selected)):
        yes_s = np.append(yes_s, [selected[i].fingerprint.numpy()], axis=0)

    tsne = TSNE(n_components=2)
    down_dim_data = tsne.fit_transform(np.concatenate([yes_s[1:], no_s[1:]], axis=0))




    sns.set_theme(style="white", font="A")
    plt.figure(figsize=(8 / 2.54, 8 / 2.54))
    sns.set_style('white')
    sns.set_style('ticks')
    sns.set_context('paper')
    fig, ax = plt.subplots()
    ax0 = ax.scatter(x=down_dim_data[101:, 0], y=down_dim_data[101:, 1], c='g', alpha=0.1)
    ax0 = ax.scatter(x=down_dim_data[:101, 0], y=down_dim_data[:101, 1], c='r')

    plt.savefig("./data/output.svg")
    plt.show()