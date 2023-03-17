import torch
from load_data import SMRTDatasetRetained,HilicDataset,MetabobaseDataset,MassBank1Dataset,Retntion_Life_Dataset_New,Retntion_Life_Dataset_Old,RikenDataset
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch import nn
from torch import optim
from typing import Iterable
from tqdm import tqdm
import random
import os
import numpy as np
import warnings
from transferDataset import TransferDataset

warnings.filterwarnings("ignore")
class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.out_lin.parameters(), lr=lr)
        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for i, data in enumerate(data_loader):
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
        total_loss = torch.tensor(0., dtype=torch.float64, device=self.device)
        with torch.no_grad():
            for data in data_loader:
                data.to(self.device)
                y_hat = self.model(data)
                total_loss += torch.abs(y_hat - data.y).sum()
        return total_loss


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
if __name__ == '__main__':
    model = torch.load('./models/best_model_download.pkl',map_location='cuda:0')
    for name, child in model.named_children():
        for param in child.parameters():
                param.requires_grad = False

    set_freeze_by_names(model,'out_lin',False)
    path = './rawFiles'
    
    batch_size = 4
    num_works = 6
    lr = 0.001
    epochs = 400
    test_batch = 4

    # dataset = MassBank1Dataset('./MassBank1Dataset/')
    listdir = os.listdir(path)
    file = open('transfer_result.txt', 'a')

    for i in range(len(listdir)):
        best_train_mae = 99999
        # best_val_mae = 99999
        best_test_mae = 99999
        if '.csv' in listdir[i]:
            join = os.path.join(path, listdir[i]).split('.csv')[0]
            # print('Loading ...')
            print(join)
            dataset = TransferDataset(join)
            train_len = int(dataset.__len__() * 0.9)
            test_len = dataset.__len__() - train_len
            set_seed(3407)
            train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trainer = Trainer(model, lr, device)
            tester = Tester(model, device)
            set_seed(3407)
            model.to(device=device)
            for epoch in range(epochs):
                if epoch%50==49:
                    trainer.optimizer.param_groups[0]['lr'] *=0.1
                print(trainer.optimizer.param_groups[0]['lr'] )
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                          num_workers=num_works, pin_memory=True,
                                          prefetch_factor=8,drop_last=True)
                # dev_loader = DataLoader(dev_data_set, batch_size=test_batch, shuffle=True,
                #                         num_workers=num_works, pin_memory=True,
                #                         prefetch_factor=8, persistent_workers=True)
                test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                                         num_workers=num_works, pin_memory=True,
                                         prefetch_factor=8,drop_last=True)
                model.train()

                loss_training = trainer.train(train_loader)

                model.eval()
                # torch.cuda.empty_cache()
                loss_train = tester.test_regressor(train_loader)
                # loss_dev = tester.test_regressor(dev_loader)
                loss_test = tester.test_regressor(test_loader)
                mae_train = loss_train.item() / ((train_len//batch_size)*batch_size)
                mae_test = loss_test.item() / ((test_len//batch_size)*batch_size)

                if mae_test < best_test_mae:
                    best_test_mae = mae_test
                    best_train_mae = mae_train

                print(f'{join}  epoch:{epoch}\ttrain_loss:{mae_train}\ttest_loss:{mae_test}')
            file.write(f'{join}    train_mae:{best_train_mae}    test_mae:{best_test_mae}\n')
            file.flush()
    file.close()