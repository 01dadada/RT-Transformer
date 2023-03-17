import argparse
import gc
import logging
import time
import timeit
import random
from typing import List

import nni
import numpy as np
from nni.utils import merge_parameter
from torch import optim

from model import MyNet
from load_data import SMRTDatasetScaffold, SMRTDatasetRetained, SMRTDataset
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split,Subset
import warnings
warnings.filterwarnings("ignore")
import os

def get_params():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--data', default='smrt', help='data folder name')
    parser.add_argument('--epochs', type=int, default=125, help='epochs')

    parser.add_argument('--batch', type=int, default=64, help='batch size')

    parser.add_argument('--works', type=int, default=8, help='num works')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--testbatch', type=int, default=2048, help='test batch size')
    parser.add_argument('--log', type=str, default='./results/results_main_model_all_data.txt', help='log file')
    parser.add_argument('--savemodel', type=str, default='best_model_all_data', help='model file name')

    parser.add_argument('--startsave', type=float, default=9999, help='test mae <= value start to save model')

    parser.add_argument('--lrdecayepoch', type=int, default=30, help='lr decay every n epoch')
    parser.add_argument('--lrdecayepochrate', type=float, default=0.1, help='lr decay every n epoch')

    parser.add_argument('--start', type=str, default='new', help='is new model?')
    parser.add_argument('--graphlayernum', type=int, default=9, help='graph layer number')
    parser.add_argument('--attentionlayernum', type=int, default=12, help='attention layer number')
    parser.add_argument('--lrsch', type=str, default='step', help='lr scheduler')
    parser.add_argument('--stepsize', type=int, default=10, help='step lr size')
    parser.add_argument('--gamma', type=float, default=0.5, help='lr gramma')
    parser.add_argument('--tmax', type=int, default=20, help='cosine T_max')
    parser.add_argument('--featdim', type=int, default=512, help='feat dim')
    parser.add_argument('--embdim', type=int, default=512, help='embedding dim')

    args, _ = parser.parse_known_args()
    return args

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def test_regressor(model, data_loader,device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data in data_loader:
            data.to(device, non_blocking=True)
            y_hat = model(data)
            # total_loss += torch.abs(y_hat - data.y).sum()
            # mre_total = torch.div(torch.abs(y_hat - data.y), data.y).sum()
            y_true.append(data.y)
            y_pred.append(y_hat)

        y_true = torch.concat(y_true)
        y_pred = torch.concat(y_pred)

        mae = torch.abs(y_true - y_pred).mean()
        # mre = torch.div(torch.abs(y_true - y_pred), y_true).mean()
        # medAE = torch.median(torch.abs(y_true - y_pred))
        # medRE = torch.median(torch.div(torch.abs(y_true - y_pred), y_true))
        #
        # score = torchmetrics.R2Score().to(self.device)
        # r2 = score(y_pred, y_true)
    # return mae.item(), medAE.item(), mre.item(), medRE.item(), r2.item()
    return mae.item()
def train_model(model, data_loader,optimizer,device):
    criterion = torch.nn.L1Loss()
    for i, data in enumerate(data_loader):
        data.to(device)
        y_hat = model(data)
        loss = criterion(y_hat, data.y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return 0

def generate_scaffolds(dataset):
    scaffolds = {}
    data_len = len(dataset)

    print("About to generate scaffolds")
    for ind, data in enumerate(dataset):
        scaffold = data.MurckoScaffold
        if scaffold not in scaffolds:
            scaffolds[scaffold] = [ind]
        else:
            scaffolds[scaffold].append(ind)

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]
    return scaffold_sets


def scaffold_split(dataset, split_size):
    train_size = split_size[0]
    test_size = split_size[1]

    scaffold_sets = generate_scaffolds(dataset)

    train_inds = []
    test_inds = []
    train_cutoff = train_size

    for scaffold_set in scaffold_sets:
        if len(train_inds) + len(scaffold_set) > train_cutoff:
            test_inds += scaffold_set
        else:
            train_inds += scaffold_set

    return [Subset(dataset,train_inds),Subset(dataset,test_inds)]
def train(args):
    feat_dim = args['featdim']
    embedding_dim = args['embdim']
    step_size = args['stepsize']
    gamma = args['gamma']
    T_max = args['tmax']
    lr_sch = args['lrsch']

    graph_layer_num = args['graphlayernum']
    attention_layer_num = args['attentionlayernum']

    num_works = args['works']
    # lr = args.lr
    # batch_size = args.batch
    lr = args['lr']
    batch_size = args['batch']
    epochs = args['epochs']
    test_batch = args['testbatch']
    seed = args['seed']
    file_name = args['log']
    save_path = f'./models/{args["savemodel"]}.pkl'
    lr_decay_epoch = args['lrdecayepoch']
    lr_decay_epoch_rate = args['lrdecayepochrate']
    mae_test_best = args['startsave']

    set_seed(1234)
    setup_seed(1234)
    if args['data'] == 'smrt':
        dataset = SMRTDataset('./SMRT')
        # dataset = SMRTDatasetScaffold('./SMRT_Scaffold')
    elif args['data'] == 'smrtretained':
        dataset = SMRTDatasetRetained('./SMRT_Retained')

    set_seed(1234)
    setup_seed(1234)

    print(len(dataset))
    dataset_len = len(dataset)
    train_len = int(dataset_len * 0.8)
    valid_len = int(dataset_len * 0.1)
    test_len = dataset_len - train_len - valid_len

    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len,valid_len, test_len])

    # train_dataset, test_valid_dataset = scaffold_split(dataset=dataset, split_size=[train_len, valid_len + test_len])
    # valid_dataset, test_dataset = scaffold_split(dataset=test_valid_dataset, split_size=[valid_len, test_len])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    train_loader2 = DataLoader(train_dataset, batch_size=test_batch, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    dev_loader = DataLoader(valid_dataset, batch_size=test_batch, shuffle=True,
                            num_workers=num_works, pin_memory=True,
                            prefetch_factor=8, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'use:', device)

    set_seed(1234)
    setup_seed(1234)
    if args['start']=='new':
        model = MyNet(emb_dim=512, feat_dim=512)
    else:
        model = torch.load(save_path)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # trainer = Trainer(model, lr, device)
    # tester = Tester(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
    #                                                      threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                      eps=1e-08)

    setup_seed(seed=seed)

    with open(file_name, 'a') as f:
        for epoch in range(epochs):
            start_time = time.time()
            if epoch % lr_decay_epoch== (lr_decay_epoch-1):
                lr *= lr_decay_epoch_rate
                print('loading model')
                model = torch.load(save_path)
                model = model.to(device)
                optimizer = optim.AdamW(model.parameters(), lr=lr)
                gc.collect()
                torch.cuda.empty_cache()

            print(optimizer.param_groups[0]['lr'])

            model.train()

            train_model(model,train_loader,optimizer,device)

            model.eval()

            mae_train = test_regressor(model,train_loader2,device)
            mae_dev = test_regressor(model,dev_loader,device)
            # scheduler.step(mae_dev)

            print(f'epoch:{epoch}    train_loss:{mae_train}    dev_loss:{mae_dev}    time:{time.time()-start_time:.2f}')
            f.write(f'epoch:{epoch}    train_loss:{mae_train}    test_loss:{mae_dev}\n')
            f.flush()

            if mae_dev < mae_test_best:
                torch.save(model, save_path)
                mae_test_best = mae_dev

            # nni.report_intermediate_result(float(mae_dev))
            # logger.debug('dev loss %g', mae_dev)
            # logger.debug('Pipe send intermediate result done.')

    # nni.report_final_result(float(mae_test_best))
    # logger.debug('Final result is %g', mae_test_best)
    # logger.debug('Send final result done.')

    model = torch.load(save_path)
    gc.collect()
    torch.cuda.empty_cache()
    final = test_regressor(model,dev_loader,device)
    print(f'{(final):.2f}')

# logger = logging.getLogger('my_AutoML')
if __name__ == '__main__':
    # try:
    #     tuner_params = nni.get_next_parameter()
    #     logger.debug(tuner_params)
    #     params = vars(merge_parameter(get_params(), tuner_params))
    #     print(params)
    #     train(params)
    # except Exception as exception:
    #     logger.exception(exception)
    #     raise
    train(vars(get_params()))







