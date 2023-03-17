import argparse
import gc
import logging
import timeit
import random
# import nni
import numpy as np
# from nni.utils import merge_parameter
from torch import optim

from model import MyNet
from load_data import SMRTDataset, SMRTDatasetRetained
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import warnings
warnings.filterwarnings("ignore")
import os

logger = logging.getLogger('my_AutoML')

def get_params():
    parser = argparse.ArgumentParser(description='Args for graph predition')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--data', default='smrt', help='data folder name')
    parser.add_argument('--epochs', type=int, default=300, help='epochs')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--works', type=int, default=8, help='num works')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--testbatch', type=int, default=2048, help='test batch size')
    parser.add_argument('--log', type=str, default='./results/results_main_model_all_data.txt', help='log file')
    parser.add_argument('--savemodel', type=str, default='best_model_all_data', help='model file name')

    parser.add_argument('--startsave', type=float, default=9999, help='test mae <= value start to save model')

    parser.add_argument('--lrdecayepoch', type=int, default=50, help='lr decay every n epoch')
    parser.add_argument('--lrdecayepochrate', type=float, default=0.1, help='lr decay every n epoch')

    # parser.add_argument('--start', type=str, default='old', help='is new model?')
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

    # set_seed(1234)
    # setup_seed(1234)
    if args['data'] == 'smrt':
        dataset = SMRTDataset('./SMRT')
    elif args['data'] == 'smrtretained':
        dataset = SMRTDatasetRetained('./SMRT_Retained')
    set_seed(1234)
    setup_seed(1234)

    train_len = int(dataset.__len__() * 0.9)
    test_len = dataset.__len__() - train_len
    train_dataset, test_dataset = random_split(dataset, [train_len, test_len])
    train_len_2 = int(train_len * 0.9)
    dev_len = train_len - train_len_2
    train_dataset, dev_dataset = random_split(train_dataset, [train_len_2, dev_len])


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    train_loader2 = DataLoader(train_dataset, batch_size=test_batch, shuffle=True,
                              num_workers=num_works, pin_memory=True,
                              prefetch_factor=8, persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=test_batch, shuffle=True,
                            num_workers=num_works, pin_memory=True,
                            prefetch_factor=8, persistent_workers=True)
    # test_loader = DataLoader(test_dataset, batch_size=test_batch, shuffle=True,
    #                          num_workers=num_works, pin_memory=True,
    #                          prefetch_factor=8, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'use\r', device)
    print('--' * 100)
    print('# of training data samples:', len(train_dataset))
    print('# of test data samples:', len(test_dataset))

    # set_seed(1234)
    # setup_seed(1234)
    if args['start']=='new':
        model = MyNet(emb_dim=512, feat_dim=512)
    else:
        model = torch.load(save_path)
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # trainer = Trainer(model, lr, device)
    # tester = Tester(model, device)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,
    #                                                      threshold=0.00001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                                      eps=1e-08)

    # setup_seed(1234)

    with open(file_name, 'a') as f:
        for epoch in range(epochs):
            if epoch % lr_decay_epoch== (lr_decay_epoch-1):
                lr *= lr_decay_epoch_rate
                print('loading model')
                model = torch.load(save_path)
                # trainer = Trainer(model, lr, device)
                # tester = Tester(model, device)
                model = model.to(device)
                optimizer = optim.Adam W(model.parameters(), lr=lr)
                # optimizer.param_groups[0]['lr'] *= lr_decay_epoch_rate
                gc.collect()
                torch.cuda.empty_cache()

            print(optimizer.param_groups[0]['lr'])

            model.train()

            train_model(model,train_loader,optimizer,device)

            model.eval()
            # loss_train = tester.test_regressor(train_loader2)
            # loss_dev = tester.test_regressor(dev_loader)
            # mae_train = loss_train.item() / train_len_2
            # mae_dev = loss_dev.item() / dev_len
            mae_train = test_regressor(model,train_loader2,device)
            mae_dev = test_regressor(model,dev_loader,device)
            # scheduler.step(mae_dev)

            print(f'epoch:{epoch}    train_loss:{mae_train}    dev_loss:{mae_dev}')
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



if __name__ == '__main__':
    try:
        # tuner_params = nni.get_next_parameter()
        # logger.debug(tuner_params)
        # params = vars(merge_parameter(get_params(), tuner_params))
        params = vars(get_params())
        print(params)
        train(params)
    except Exception as exception:
        logger.exception(exception)
        raise

