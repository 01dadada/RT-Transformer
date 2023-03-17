import os
from tqdm import tqdm
import multiprocessing
from rdkit.Chem.inchi import *
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import urllib
import pandas as pd
import numpy as np
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch.utils.data import random_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool
from functools import partial
from rdkit.Chem.inchi import *
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import urllib
import pandas as pd
import numpy as np
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch.utils.data import random_split
import warnings
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

def ROC(df, threshold):
    # TPR=[]
    # FPR=[]
    TPR=[0]*len(threshold)
    FPR=[0]*len(threshold)

    df['pos'] = df['inchi']==df['inchi_true']
    P = len(df[df.pos==True])
    N = len(df[df.pos==False])
    print(P, N)
    for index, thresh in enumerate(tqdm(threshold)):
        df['pos_thresh'] = df['re']<thresh
        TP = len(df[(df['pos']==True)&(df['pos_thresh']==True)])
        FP = len(df[(df['pos']==False)&(df['pos_thresh']==True)])
        TPR[index] = TP/P
        FPR[index] = FP/N
        # TPR.append(TP/P)
        # FPR.append(FP/N)
    print ('Best threshold:', threshold[np.argmax(np.array(TPR)-np.array(FPR))])
    return (TPR, FPR, threshold[np.argmax(np.array(TPR)-np.array(FPR))])
def ROC_AUC(TPR, FPR):
    ROC_AUC=0
    for i in tqdm(range(len(FPR)-1)):
        ROC_AUC+= (FPR[i+1]-FPR[i])*(TPR[i+1]+TPR[i])/2
    return ROC_AUC

def get_search_space_reduction(df, threshold):
    df['filtered'] = df['re']>threshold
    total, filtered=[0]*len(df[df['pos']==True].inchi),[0]*len(df[df['pos']==True].inchi)
    for index, inchi in enumerate(tqdm(df[df['pos']==True].inchi)):
        new_df = df[df['inchi_true']==inchi]
        total[index] = len(new_df)
        filtered[index] = len(new_df[new_df['filtered'] ==True])
        # total.append(len(new_df))
        # filtered.append(len(new_df[new_df['filtered'] ==True]))
    return total, filtered

if __name__ == '__main__':
    path = './compare'
    listdir = os.listdir(path)
    datasets_names = []
    for name in listdir:
        if not '.csv' in name:
            datasets_names.append(name)
    datasets_names.sort()
    
    for dataset_name in datasets_names:
        
        frame = pd.read_csv(f'./results/{dataset_name}_results.csv')
        frame['pos'] = frame['inchi']==frame['inchi_true']
        frame['re'] = 100*abs(frame['rt_true']-frame['rt_pred'])/frame['rt_true']
        
        threshold=np.arange(0, 200, 0.01)
        TPR, FPR, thresh = ROC(frame, threshold)
        AUC = ROC_AUC(TPR, FPR)
        total, filtered = get_search_space_reduction(frame,thresh)
        mean = np.divide(filtered,total).mean()
        print(f'{dataset_name}的AUC:{AUC}')
        print(f'{dataset_name}平均能够过滤{mean*100}%的候选分子')
        
        x=[0,1]
        y=[0,1]
        labels=['Total', 'Filtered']
        fig, axs = plt.subplots(1,2, figsize=(6, 3.5), sharey=False, dpi=1200)
        i=0
        matplotlib.rcParams.update({'font.size': 8})

        axs[0].plot(FPR, TPR)
        axs[0].plot(x,y)
        axs[0].set_yticks([0,0.5, 1])
        axs[0].set_xticks([0,0.5, 1])
        axs[0].set_xlabel('FPR')
        axs[0].set_ylabel('TPR')
        axs[0].set_yticklabels(['0', '0.5', '1'])
        axs[0].set_xticklabels(['0', '0.5', '1'])
        axs[0].set_title('ROC')

        axs[1].boxplot((total, filtered), showfliers=False)
        axs[1].set_xticklabels(labels, fontsize=8)
        axs[1].set_ylabel('No candidates')
        axs[1].set_title('filter capacity')

        fig.suptitle('Eawag_XBridgeC18')
        fig.tight_layout(pad=3.0)
        plt.savefig(f'./figs/{dataset_name}.svg')
        
        
