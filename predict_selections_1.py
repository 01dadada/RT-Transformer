import numpy as np
import torch
from torch_geometric.loader import DataLoader, DataListLoader
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
from transferDataset import PredictionDataset
from tqdm import tqdm

# from multiprocessing import Pool
# import multiprocessing
# multiprocessing.set_start_method('spawn')
import os
from multiprocessing import cpu_count



if __name__ == '__main__':

    cuda_num = 'cuda:1'
    device = torch.device(cuda_num if torch.cuda.is_available() else 'cpu')
    print(f'use:', device)
    torch.manual_seed(1234)
    model = torch.load('./models/best_model.pkl', map_location=cuda_num)
    model.to(device=device)
    model.eval()

    np.random.seed(1234)


    # path = './search_smrt_retained/'
    # save_dir = './results_of_search_smrt_retained/'
    path = './search_smrt_all/'
    save_dir = './results_of_search_smrt_all/'
    # path = './search_smrt'
    # save_dir = './results_of_search_smrt'
    listdir = os.listdir(path)
    files = []
    for filename in listdir:
        if filename.endswith('.csv') and not os.path.exists(save_dir+'/'+filename.split('.csv')[0]+'_results.csv'):
            files.append(filename)

    print(len(files))
    files.sort()
    files = files[-5000:-2000]
    for filename in tqdm(files, ncols=50):
        # print(os.path.join(path, filename).split('./')[1].split('.')[0])
        # print(os.path.join(path, filename).split('./')[1].split('.')[0])
        pred_data = PredictionDataset(os.path.join(path, filename).split('./')[1].split('.')[0])
        loader = DataLoader(pred_data,
                            batch_size=4096,
                            shuffle=True,
                            num_workers=8,
                            pin_memory=True,
                            prefetch_factor=1024,
                            persistent_workers=True,
                            )
        result = pd.DataFrame(index=range(100000),columns=['inchi', 'pred_rt'])
        index = 0

        with torch.no_grad():
            for data in loader:
                data.to(device)
                y_hat = model(data)
                y_hat = y_hat.reshape(-1)
                if len(y_hat) > 1:
                    for i in range(len(y_hat)):
                        result.loc[index] = {'inchi': data[i].inchi, 'pred_rt': y_hat[i].cpu().item()}
                        index += 1
                else:
                    result.loc[index] = {'inchi': data[0].inchi, 'pred_rt': y_hat.cpu().item()}
                    index += 1
        result = result.dropna()
        formula = filename.split('.')[0]
        save_path = os.path.join(save_dir, formula) + '_results.csv'
        result.to_csv(save_path, index=False)
        # print(f'保存成功{save_path}')

    print('测试结束')