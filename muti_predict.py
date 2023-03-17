import numpy as np
import torch
from torch_geometric.loader import DataLoader, DataListLoader
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from transferDataset import PredictionDataset
from tqdm import tqdm
import torch.multiprocessing as mp
# from multiprocessing import Pool
import multiprocessing
# multiprocessing.set_start_method('forkserver', force=True)
import os
import random
import shutil
from multiprocessing import cpu_count
warnings.filterwarnings("ignore")

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)





# save_dir = './results_of_search_smrt_all'
# path = './search_5_datasets'
path = './search_smrt_all'
# path = './search_smrt_all'


def train(filename):
    try:
        
        pred_data = PredictionDataset(os.path.join(path, filename).split('./')[1].split('.')[0])
    except:
        # print(filename)
        pass
        # shutil.rmtree(filename.split('.')[0])
if __name__ == '__main__':
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # print(f'use:', device)
    # torch.manual_seed(1234)
    # # model = torch.load('./models/best_model_download.pkl', map_location='cuda:0')
    # # model.to(device=device)
    # # model.eval()

    # np.random.seed(1234)
    set_seed(1234)

    listdir = os.listdir(path)

    files = []
    for filename in listdir:
        if filename.endswith('.csv'):
            files.append(filename)
    files.sort()
    print(len(files))
    files = files[-500:]
    # def train(filename):
    #     pred_data = PredictionDataset(os.path.join(path, filename).split('./')[1].split('.')[0])

    ctx = torch.multiprocessing.get_context("spawn")
    pool_list = []
    num_processes = 50
    pool = ctx.Pool(num_processes)

    pool.map(train, files)

    pool.close()
    pool.join()

    print('测试结束')