import gc
import time
import numpy
import torch
from model import Trainer
from load_data import SMRTDatasetRetained, HilicDataset, MetabobaseDataset, MassBank1Dataset, Retntion_Life_Dataset_New, \
    Retntion_Life_Dataset_Old, RikenDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
# from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from typing import Iterable
from tqdm import tqdm
import random
import os
import numpy as np
import warnings
from transferDataset import TransferDataset
from sklearn.metrics import r2_score
import torchmetrics
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GAE, GATv2Conv, GraphSAGE, GENConv, GMMConv, \
    GravNetConv, MessagePassing, global_max_pool, global_add_pool, GAT, GINConv, GINEConv, GraphNorm, SAGEConv, RGATConv
from torch_geometric.nn import global_sort_pool
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool, MessagePassing
from torch_geometric.utils import add_self_loops
from tqdm import tqdm
from torch.nn import Conv1d
from model import GraphTransformerBlock ,OneDimConvBlock

warnings.filterwarnings("ignore")

class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for data in data_loader:
            data.to(self.device)
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
                y_pred.append(y_hat.reshape([-1]))

            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)

            mae = torch.abs(y_true - y_pred).mean()
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))

            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)
        return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()


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


def set_freeze(model):
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
import os
from multiprocessing import cpu_count
import torch.multiprocessing



if __name__ == '__main__':
    kfold = 10
    batch_size = 8
    num_works = 4
    lr = 0.01
    epochs = 300
    test_batch = 8
    seeds = [1234,12345,123456,1234567,12345678]
    # seeds = [1234]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # path = './rawFiles'
    # path = './compare'
    path = './Riken'
    listdir = os.listdir(path)


    listdir.sort()
    file = open('./results/temp.csv', 'a')
    file.write(f'seed,dataset,fold,mae,mre,medAE,medRE,r2\n')
    file.write(f'lr:{lr},batch_size:{batch_size}\n')
    file.flush()

    for i in range(len(listdir)):
        if '.csv' in listdir[i]:
            join = os.path.join(path, listdir[i]).split('.csv')[0]
        else:
            continue
        print(join)
        for seed in seeds:
            for fold in range(kfold):
                new_lr = lr
                dataset_name = join.split('/')[2]
                dataset = TransferDataset(join)
                fold_size = len(dataset) // kfold
                fold_reminder = len(dataset) % kfold
                split_list = [fold_size] * kfold
                for reminder in range(fold_reminder):
                    split_list[reminder] = split_list[reminder] + 1
                set_seed(seed)
                split = random_split(dataset, split_list)
                best_test_mae = 9999999
                set_seed(seed)
                model = torch.load('./models/best_model.pkl',map_location="cuda:0")
                set_freeze(model=model)
                set_freeze_by_names(model, 'out_lin', False)
                model.to(device=device)
                trainer = Trainer(model, new_lr, device)
                tester = Tester(model, device)
                gc.collect()
                torch.cuda.empty_cache()
                test_dataset = split[fold]
                train_list = []
                for m in range(kfold):
                    if m != fold:
                        train_list.append(split[m])
                train_dataset = torch.utils.data.ConcatDataset(train_list)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_works, pin_memory=True,
                                          prefetch_factor=4, drop_last=True,)
                test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                                         num_workers=num_works, pin_memory=True,
                                         prefetch_factor=4, drop_last=True,)
                # train_loader = list(train_loader)
                # test_loader = list(test_loader)
                time_start = time.time()
                for epoch in tqdm(range(epochs),ncols=50):
                    if epoch % 50 == 49:
                        # trainer.optimizer.param_groups[0]['lr'] *= 0.1
                        model = torch.load( f'./models/{dataset_name}/fold{fold}.pkl')
                        new_lr = new_lr*0.1
                        trainer = Trainer(model=model,lr=new_lr,device=device)
                        tester = Tester(model=model,device=device)
                    model.train()
                    trainer.train(train_loader)
                    model.eval()
                    # mae_train, medAE_train, mre_train, medRE_train, r2_train = tester.test_regressor(train_loader)
                    mae_test, medAE_test, mre_test, medRE_test, r2_test = tester.test_regressor(test_loader)
                    if mae_test < best_test_mae:
                        best_test_mae = mae_test
                        best_test_medAE = medAE_test
                        best_test_mre = mre_test
                        best_test_medRE = medRE_test
                        r2_test_best = r2_test
                        torch.save(model, f'./models/{dataset_name}/fold{fold}.pkl')

                print(f'time:{time.time()-time_start:.2f},dataset:{dataset_name},fold:{fold},mae:{best_test_mae:.2f},mre:{best_test_mre * 100:.2f},medAE:{best_test_medAE:.2f},medRE:{best_test_medRE * 100:.2f},r2:{r2_test_best:.2f}')
                file.write(f'{seed},{dataset_name},{fold},{best_test_mae},{best_test_mre * 100},{best_test_medAE:},{best_test_medRE * 100},{r2_test_best}\n')
                file.flush()
    file.close()
