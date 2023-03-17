import timeit
import numpy as np
from model import MyNet, Trainer
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import torchmetrics


warnings.filterwarnings("ignore")



class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def test_regressor(self, data_loader):

        y_true = [0]*len(data_loader)*8
        y_pred = [0]*len(data_loader)*8
        with torch.no_grad():
            for index, data in enumerate(data_loader):
                data.to(self.device)
                y_hat = self.model(data)
                y_true[index*8:index*8+8] = data.y
                y_pred[index*8:index*8+8] = y_hat
                # y_true.append(data.y)
                # y_pred.append(y_hat)

            y_true = torch.tensor(y_true)
            # numpy.expand_dims(y_true)
            y_pred = torch.tensor(y_pred)

            mae = torch.abs(y_true - y_pred).mean()
            mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
            medAE = torch.median(torch.abs(y_true - y_pred))
            medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))

            score = torchmetrics.R2Score().to(self.device)
            r2 = score(y_pred, y_true)
        return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    batch_size = 8
    num_works = 8
    test_batch =8

    torch.manual_seed(1234)
    # dataset = SMRTDataset('./SMRT')
    dataset = SMRTDatasetRetained('./SMRT_Rertained')
    # dataset = MassBank1Dataset('./MassBank1Dataset/')
    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_len_2 = int(train_len*0.9)
    dev_len = train_len-train_len_2
    train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    dev_loader = DataLoader(dev_data_set, batch_size=test_batch, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
                             num_workers=num_works, pin_memory=True,
                             prefetch_factor=8, persistent_workers=True)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f'use\r', device)
    torch.manual_seed(1234)
    # model = MyNet(emb_dim=512, feat_dim=512)
    model = \
        torch.load('models/best_model_download.pkl', map_location='cuda:1')
    tester = Tester(model, device)
    np.random.seed(1234)

    model.to(device=device)

    model.eval()
    # # loss_train = tester.test_regressor(train_loader)
    # loss_dev = tester.test_regressor(dev_loader)
    # loss_test = tester.test_regressor(test_loader)
    mae, medAE, mre, medRE, r2= tester.test_regressor(test_loader)
    # # mae_train = loss_train.item() / train_len_2
    # mae_dev = loss_dev.item() / dev_len
    # mae_test = loss_test.item() / test_len
    # print(mae_train,mae_dev,mae_test)
    # print(mae_dev,mae_test)
    print(f"maeP{mae:.2f}, medAE{medAE:.2f}, mre{mre*100:.2f}, medRE{medRE*100:.2f}, r2{r2*100:.2f}")


