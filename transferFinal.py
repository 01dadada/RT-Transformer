import gc
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from typing import Iterable
import random
import warnings
from transferDataset import TransferDataset
from sklearn.metrics import r2_score
import torchmetrics
import torch
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore")
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

import os
from multiprocessing import cpu_count
import torch.multiprocessing



if __name__ == '__main__':

    kfold = 10
    batch_size = 8
    num_works = 4
    lr = 0.1
    epochs = 400
    test_batch = 8
    seeds = [1234,12345,123456,1234567,12345678]
    # seeds = [1234]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = './Riken'
    # path = './compare'
    listdir = os.listdir(path)

    listdir.sort()
    # file = open('./results/Full_data_transfer_all_seeds_compare_1.csv', 'a')
    file = open('./results/temp.csv', 'a')
    file.write(f'seed,dataset,fold,mae,mre,medAE,medRE,r2\n')
    file.flush()

    for i in range(len(listdir)):
        if '.csv' in listdir[i]:
            join = os.path.join(path, listdir[i]).split('.csv')[0]
        else:
            continue

        dataset = TransferDataset(join)
        
        dataset_name = join.split('/')[2]

        for seed in seeds:
            fold_size = len(dataset) // kfold
            fold_reminder = len(dataset) % kfold
            split_list = [fold_size] * kfold
            for reminder in range(fold_reminder):
                split_list[reminder] = split_list[reminder] + 1
            set_seed(seed)
            split = random_split(dataset, split_list)
            
            for fold in range(kfold):
                best_train_mae = 99999
                best_test_mae = 99999
                set_seed(seed)
                test_dataset = split[fold]
                train_list = []
                for m in range(kfold):
                    if m != fold:
                        train_list.append(split[m])
                train_dataset = torch.utils.data.ConcatDataset(train_list)

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_works, pin_memory=True,
                                          prefetch_factor=2, drop_last=True,persistent_workers=True)
                test_loader = DataLoader(test_dataset, batch_size=test_batch,
                                         num_workers=num_works, pin_memory=True,
                                         prefetch_factor=2,persistent_workers=True,)
                set_seed(seed)
                model = torch.load('./models/best_model.pkl')
                # model = torch.load('./models/best_model.pkl', map_location='cuda:0')
                for name, child in model.named_children():
                    for param in child.parameters():
                        param.requires_grad = False
                set_freeze_by_names(model, 'out_lin', False)
                model.to(device=device)
                trainer = Trainer(model, lr, device)
                tester = Tester(model, device)

                gc.collect()
                torch.cuda.empty_cache()

                set_seed(seed)
                for epoch in tqdm(range(epochs),ncols=50):
                    if epoch % 50 == 49:
                        trainer.optimizer.param_groups[0]['lr'] *= 0.1
                    model.train()
                    loss_training = trainer.train(train_loader)
                    model.eval()
                    # mae_train, medAE_train, mre_train, medRE_train, r2_train = tester.test_regressor(train_loader)
                    mae_test, medAE_test, mre_test, medRE_test, r2_test = tester.test_regressor(test_loader)
                    if mae_test < best_test_mae:
                        best_test_mae = mae_test
                        best_test_medAE = medAE_test
                        best_test_mre = mre_test
                        best_test_medRE = medRE_test
                        r2_test_best = r2_test
                        # torch.save(model, f'./models/{dataset_name}/fold{fold}.pkl')

                print(f'dataset:{dataset_name},fold:{fold},mae:{best_test_mae:.2f},mre:{best_test_mre * 100:.2f},medAE:{best_test_medAE:.2f},medRE:{best_test_medRE * 100:.2f},r2:{r2_test_best:.2f}')
                file.write(f'{seed},{dataset_name},{fold},{best_test_mae},{best_test_mre * 100},{best_test_medAE:},{best_test_medRE * 100},{r2_test_best}\n')
                file.flush()
        break
    file.close()
