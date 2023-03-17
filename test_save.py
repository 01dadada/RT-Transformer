import timeit
import numpy as np
from model import MyNet, Trainer,Tester
from load_data import SMRTDataset, SMRTDatasetRetained, RikenDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    class Tester(object):
        def __init__(self, model, device):
            self.model = model
            self.device = device

        def test_regressor(self, data_loader):
            total_loss = torch.tensor(0, dtype=torch.float64, device=self.device)
            with torch.no_grad():
                for data in tqdm(data_loader):
                    data.to(self.device)
                    y_hat = self.model(data)
                    total_loss += torch.abs(y_hat - data.y).sum()
            return total_loss

        def test_regressor_save(self, data_loader, df):
            total_loss = torch.tensor(0, dtype=torch.float64, device=self.device)
            index = 0
            with torch.no_grad():
                for i, data in tqdm(enumerate(data_loader)):
                    data.to(self.device)
                    y_hat = self.model(data)
                    for j, y_pre in enumerate(y_hat):
                        df.loc[index] = {'y':data.y[j].cpu().item(),'y_pred':y_pre.cpu().item()}
                        index += 1
                    total_loss += torch.abs(y_hat - data.y).sum()
            return total_loss
    batch_size = 1024
    num_works = 4
    test_batch = 1024

    torch.manual_seed(1234)
    # dataset = SMRTDatasetRetained('./SMRT_Retain')
    dataset = SMRTDataset('./SMRT')
    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_len_2 = int(train_len*0.9)
    dev_len = train_len-train_len_2
    train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=16, persistent_workers=True)
    dev_loader = DataLoader(dev_data_set, batch_size=test_batch, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                             num_workers=num_works, pin_memory=True,
                             prefetch_factor=16, persistent_workers=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use:', device)
    torch.manual_seed(1234)
    # model = MyNet(emb_dim=512, feat_dim=512)
    # model = torch.load('./models/best_model_download.pkl',map_location='cuda:0')
    model = torch.load('./models/best_model.pkl',map_location='cuda:0')
    tester = Tester(model, device)
    np.random.seed(1234)

    model.to(device=device)
    df = pd.DataFrame()
    df.insert(loc=0, column='y', value=0)
    df.insert(loc=0, column='y_pred', value=0)

    model.eval()
    loss_train = tester.test_regressor(train_loader)
    loss_dev = tester.test_regressor(dev_loader)
    loss_test = tester.test_regressor_save(test_loader,df)
    
    # df.to_csv('./results/pred_retained.csv')
    df.to_csv('./results/pred_all.csv')
    
    mae_train = loss_train.item() / train_len_2
    mae_dev = loss_dev.item() / dev_len
    mae_test = loss_test.item() / test_len
    print(mae_train,mae_dev,mae_test)


