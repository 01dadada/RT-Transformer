import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from model import MyNet, Trainer, Tester
from load_data import SMRTDataset, SMRTDatasetRetained
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
import random
import os

warnings.filterwarnings("ignore")


class Trainer(object):
    def __init__(self, model, lr, device):
        self.model = model
        from torch import optim
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, 30)
        self.device = device

    def train(self, data_loader):
        criterion = torch.nn.L1Loss()
        for i, data in enumerate(tqdm(data_loader)):
            data.to(self.device)
            y_hat = self.model(data)
            loss = criterion(y_hat, data.y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

        return 0
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    batch_size = 64
    num_works = 8
    lr = 0.00001
    epochs = 500
    test_batch = 2048

    torch.manual_seed(1234)
    set_seed(1234)
    dataset = SMRTDataset('./SMRT')
    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

    train_len_2 = int(train_len * 0.9)
    dev_len = train_len - train_len_2
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'use\r', device)
    print('-' * 100)
    print('The preprocess has finished!')
    print('# of training data samples:', len(train_dataset))
    print('# of test data samples:', len(test_dataset))
    print('-' * 100)
    print('Creating a model.')
    torch.manual_seed(1234)
    model = MyNet(emb_dim=512, feat_dim=512)
    # model = torch.load('./model/best_model2.pkl', map_location='cuda:0')
    # model = torch.load('best_model_epoch102_loss_32.12735703089935.pkl',map_location='cuda:0')
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear):
    #         torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    trainer = Trainer(model, lr, device)
    tester = Tester(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-' * 100)
    print('Start training.')
    print('The result is saved in the output directory every epoch!')

    np.random.seed(1234)
    torch.manual_seed(1234)

    model.to(device=device)
    mae_test_best = 40
    with open('./results/try.txt', 'a') as f:
        for epoch in range(epochs):
            print(trainer.optimizer.param_groups[0]['lr'])
            model.train()

            loss_training = trainer.train(train_loader)

            model.eval()
            # torch.cuda.empty_cache()
            loss_train = tester.test_regressor(train_loader)
            loss_dev = tester.test_regressor(dev_loader)
            mae_train = loss_train.item() / train_len_2
            mae_dev = loss_dev.item() / dev_len

            print(f'epoch:{epoch}\ttrain_loss:{mae_train}\ttest_loss:{mae_dev}')
            f.write(f'epoch:{epoch}\ttrain_loss:{mae_train}\ttest_loss:{mae_dev}\n')
            f.flush()

            if mae_dev < mae_test_best:
                # torch.save(model, f'./model/best_model_epoch{epoch}_loss_{mae_dev}.pkl')
                torch.save(model, f'./model/best_model2.pkl')
                mae_test_best = mae_dev

    model = torch.load('./model/best_model2.pkl', map_location='cuda:0')
    tester.test_regressor(dev_loader)

