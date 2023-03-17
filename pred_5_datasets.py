import gc

import pandas as pd
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Iterable
import random
import warnings
from transferDataset import TransferDataset, PredictionDataset
from sklearn.metrics import r2_score
import torchmetrics
import torch
import numpy as np
from tqdm import tqdm
import os

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            param.requires_grad = not freeze


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

import os
from multiprocessing import cpu_count
import torch.multiprocessing
class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for i, data in enumerate(data_loader):
            data.to(self.device,non_blocking=True)
            y_hat = self.model(data)
            loss = criterion(y_hat, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return 0
class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_regressor(self, data_loader):
        y_true = []
        y_pred = []
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device,non_blocking=True)
                y_hat = self.model(data)
                # total_loss += torch.abs(y_hat - data.y).sum()
                # mre_total = torch.div(torch.abs(y_hat - data.y), data.y).sum()
                y_true.append(data.y)
                y_pred.append(y_hat)

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            mae = torch.abs(y_true - y_pred).mean()
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))

            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)
        return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()


if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy('file_system')
    # cpu_num = 4
    # # 自动获取最大核心数目
    # print(cpu_num)
    # os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    # os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    # os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    # os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    # os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    # torch.set_num_threads(cpu_num)
    # torch.backends.cudnn.benchmark = True

    kfold = 10
    seeds = [1234]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = './compare'
    listdir = os.listdir(path)

    listdir.sort()
    seed = 1234
    for i in range(len(listdir)):
        if '.csv' in listdir[i]:
            join = os.path.join(path, listdir[i]).split('.csv')[0]
        else:
            continue
        
        set_seed(seed)
        dataset = TransferDataset(join)
        print(join)
        dataset_name = join.split('/')[2]
        # if dataset_name < 'Eb':
        #     continue
        fold_size = len(dataset) // kfold
        fold_reminder = len(dataset) % kfold
        split_list = [fold_size] * kfold
        for reminder in range(fold_reminder):
            split_list[reminder] = split_list[reminder] + 1
        set_seed(seed)
        split = random_split(dataset, split_list)
        frame = pd.DataFrame(index=range(len(dataset)*10000),columns=['inchi_true', 'rt_true', 'inchi', 'rt_pred'])
        
        frame_index = 0
        for fold in range(kfold):
            model = torch.load(f'./models/{dataset_name}/fold{fold}.pkl')
            set_seed(seed)
            test_dataset = split[fold]
            train_list = []
            for m in range(kfold):
                if m != fold:
                    train_list.append(split[m])
            train_dataset = torch.utils.data.ConcatDataset(train_list)

            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True,
                                     num_workers=4, pin_memory=False,
                                     prefetch_factor=4, drop_last=False,
                                     persistent_workers=True, )
            test_loader = list(test_loader)

            for data in tqdm(test_loader, ncols=50):
                inchi_true = data.inchi
                inchi_true = str(inchi_true).split('[')[1].split(']')[0].replace("'","")
                rt_true = data.y.cpu().item()
                formula = data.formula[0]

                next_data_path = f'./search_5_datasets/{formula}'
                predict_dataset = PredictionDataset(next_data_path)
                predict_loader = DataLoader(predict_dataset,
                                            batch_size=1024,
                                            shuffle=True,
                                            num_workers=4,
                                            pin_memory=False,
                                            prefetch_factor=512,
                                            persistent_workers=True)

                for data_for_pred in predict_loader:
                    data_for_pred.to(device)
                    pred = model(data_for_pred)
                    reshape = pred.reshape(-1)
                    for z in range(len(reshape)):
                        pred_rt = reshape[z]
                        inchi = data_for_pred[z].inchi
                        frame.loc[frame_index] = {
                            'inchi_true': inchi_true,
                            'rt_true': rt_true,
                            'inchi': inchi,
                            'rt_pred': pred_rt.cpu().item()
                        }
                        frame_index = frame_index + 1
        frame = frame.dropna()
        frame.to_csv(f'results/{dataset_name}_results.csv')