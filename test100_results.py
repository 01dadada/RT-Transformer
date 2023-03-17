import numpy as np
from model import MyNet, Trainer, Tester
from load_data import SMRTDataset, SMRTDatasetRetained
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from tqdm import tqdm

threshold = 60
torch.manual_seed(1234)
dataset = SMRTDatasetRetained('./SMRT_Retain')
train_len = int(dataset.__len__() * 0.9)
test_len = dataset.__len__() - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_len_2 = int(train_len * 0.9)
dev_len = train_len - train_len_2
train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])
selected, no_selected = random_split(test_dataset, [100, len(test_dataset) - 100])

frame = pd.DataFrame(columns=[
                              # 'inchi',
                              'formula',
                              'experimental RT',
                              'Num of candidates searched from PubChem',
                              'Num of candidates filtered by RT-Transformer',
                              'Rate of candidates filtered',
                              'candidate is filtered out'
                              ])
for data in selected:
    read_csv = pd.read_csv(os.path.join('search_form_pubchem', f'{data.formula}_results.csv'))
    filter_num = (abs(read_csv['pred_rt'] - data.y.cpu().numpy()) > threshold).value_counts()[True]
    all_num = len(read_csv)
    is_select = False
    inchis = read_csv[abs(read_csv['pred_rt'] - data.y.cpu().numpy()) <= threshold]['inchi']
    for inchi in inchis:
        if data.inchi == inchi:
            print('成功筛选')
            is_select = True
    series = pd.Series(
        {
            # 'inchi':data.inchi,
            'formula': data.formula,
            'experimental RT': data.y.cpu().numpy().item(),
            'Num of candidates searched from PubChem': all_num,
            'Num of candidates filtered by RT-Transformer': filter_num,
            'Rate of candidates filtered': filter_num / all_num * 100,
            'candidate is filtered out': is_select
        }
    )
    frame = frame.append(series,ignore_index=True)
frame.to_csv('results/100molecules_result.csv')





