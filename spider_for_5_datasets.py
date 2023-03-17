from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch.utils.data import random_split
import warnings
import os
import time
import requests
import urllib
from multiprocessing import Pool
from rdkit import Chem
from transferDataset import TransferDataset, PredictionDataset


def get_isomers(formula):
    start_time = time.time()
    status_code = 0
    # print(formula,end=' ')
    while status_code < 200 or status_code >= 300:
        get = requests.get(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/formula/' + formula + '/property/InChI/txt')
        listkey = str(get.content).split('ListKey: ')[1].rstrip("'")
        status_code = get.status_code
    status_code = 0
    # print('查询listkey成功！ ', end=' ')
    while status_code < 200 or status_code >= 300:
        get_2 = requests.get(
            'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/listkey/' + str(listkey) + '/property/InChI/txt')
        if 'InChI' in get_2.text:
            with open(f'./search_5_datasets/{formula}.csv', 'w+', encoding='utf-8') as f:
                f.write(get_2.text)
            status_code = get_2.status_code
    print(f'写入文件成功{formula}.csv', end=' ')
    print(f'花费时间{(time.time() - start_time):.2f}')


if __name__ == '__main__':
    torch.manual_seed(1234)
    path = './compare'
    # path = './compare'
    listdir = os.listdir(path)

    listdir.sort()

    for i in range(len(listdir)):
        if '.csv' in listdir[i]:
            join = os.path.join(path, listdir[i]).split('.csv')[0]
        else:
            continue

        # set_seed(seed)
        dataset = TransferDataset(join)
        print(join)
        dataset_name = join.split('/')[2]

        test_dataset = list(dataset)
        formula_list = []
        for i in range(len(test_dataset)):
            if not os.path.exists(f'./search_5_datasets/{str(test_dataset[i].formula)}.csv'):
                inchi = test_dataset[i].inchi
                mol = None
                try:
                    mol = Chem.MolFromInchi(inchi)
                except:
                    print('转化失败')
                if mol == None:
                    continue
                formula_list.append(str(test_dataset[i].formula))
        formula_list = list(set(formula_list))

        print(len(formula_list))
        # if len(formula_list)>0:
        #     print(formula_list)
        pool = Pool(12)
        pool.map(get_isomers, formula_list)

        pool.close()
        pool.join()

        # for data in test_dataset:
        #     formula = data.formula
        #     get_isomers(formula)
