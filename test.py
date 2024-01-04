import timeit
import numpy as np
from torch_geometric import io

from model import MyNet, Trainer, Tester
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
from model import MyNet

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    batch_size = 1024
    num_works = 8
    test_batch = 1024

    torch.manual_seed(1234)
    dataset = SMRTDatasetRetained('./SMRT_Retained')
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
    print(f'use\r', device)
    torch.manual_seed(1234)

    
    # model = torch.load('./models/best_model_download.pkl', map_location='cpu')
    model = MyNet(emb_dim=512, feat_dim=512)
    state = torch.load('./best_state_download_dict.pth')
    model.load_state_dict(state)
    model.eval()

    tester = Tester(model, device)
    np.random.seed(1234)

    model.to(device=device)

    model.eval()
    loss_train = tester.test_regressor(train_loader)
    loss_dev = tester.test_regressor(dev_loader)
    loss_test = tester.test_regressor(test_loader)
    mae_train = loss_train.item() / train_len_2
    mae_dev = loss_dev.item() / dev_len
    mae_test = loss_test.item() / test_len
    print(mae_train,mae_dev,mae_test)


