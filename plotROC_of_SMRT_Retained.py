import pandas as pd
import numpy as np
from load_data import SMRTDataset, SMRTDatasetRetained, MetabobaseDataset, RikenDataset, MassBank1Dataset
import torch
from torch.utils.data import random_split
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
torch.manual_seed(1234)
dataset = SMRTDatasetRetained('./SMRT_Retain')
train_len = int(dataset.__len__() * 0.9)
test_len = dataset.__len__() - train_len
train_dataset, test_dataset = random_split(dataset, [train_len, test_len])

train_len_2 = int(train_len * 0.9)
dev_len = train_len - train_len_2
train_dataset, dev_data_set = random_split(train_dataset, [train_len_2, dev_len])
selected, no_selected = random_split(test_dataset, [100, len(test_dataset) - 100])

test_data_frame = pd.DataFrame(columns=['inchi', 'formula','real_rt'])

for i in range(len(selected)):
    data = selected[i]
    test_data_frame.loc[i] = {
        'inchi':data.inchi,
        'formula':data.formula,
        'real_rt':data.y.cpu().item()
    }

frame = pd.DataFrame(columns=['inchi_true','rt_true','inchi','rt_pred'])
for i in range(len(test_data_frame)):
    inchi_true = test_data_frame['inchi'][i]
    rt_true = test_data_frame['real_rt'][i]
    formula = test_data_frame['formula'][i]

    csv = pd.read_csv(f'./search_form_pubchem/{formula}_results.csv')
    index = 0
    for j in range(len(csv)):
        frame.loc[index] = {
                                    'inchi_true':inchi_true,
                                    'rt_true':rt_true,
                                    'inchi':csv['inchi'][j],
                                    'rt_pred':csv['pred_rt'][j]
                            }
        index = index + 1

frame['re'] = 100*abs(frame['rt_true']-frame['rt_pred'])/frame['rt_true']

def ROC(df, threshold):
    TPR=[]
    FPR=[]

    df['pos'] = df['inchi']==df['inchi_true']
    P = len(df[df.pos==True])

    N = len(df[df.pos==False])
    print(P, N)
    for thresh in threshold:
        df['pos_thresh'] = df['re']<thresh
        TP = len(df[(df['pos']==True)&(df['pos_thresh']==True)])
        FP = len(df[(df['pos']==False)&(df['pos_thresh']==True)])
        TPR.append(TP/P)
        FPR.append(FP/N)
    print ('Best threshold:', threshold[np.argmax(np.array(TPR)-np.array(FPR))])
    return (TPR, FPR, threshold[np.argmax(np.array(TPR)-np.array(FPR))])
def ROC_AUC(TPR, FPR):
    ROC_AUC=0
    for i in range(len(FPR)-1):
        ROC_AUC+= (FPR[i+1]-FPR[i])*(TPR[i+1]+TPR[i])/2
    return ROC_AUC

threshold=np.arange(0, 200, 0.01)
TPR, FPR, thresh = ROC(frame, threshold)
AUC = ROC_AUC(TPR, FPR)


def get_search_space_reduction(df, threshold):
    df['filtered'] = df['re']>threshold
    total, filtered=[],[]
    for inchi in df[df['pos']==True].inchi:
        new_df = df[df['inchi_true']==inchi]
        total.append(len(new_df))
        filtered.append(len(new_df[new_df['filtered'] ==True]))
    return total, filtered

total, filtered = get_search_space_reduction(frame,thresh)


import matplotlib
x=[0,1]
y=[0,1]
labels=['Total', 'Filtered']
fig, axs = plt.subplots(1,2, figsize=(6, 3.2), sharey=False, dpi=1200)
i=0
matplotlib.rcParams.update({'font.size': 8})
# for ds in [ds_name]:
    #print(st.mean(RF[ds]))
axs[0].plot(FPR, TPR)
axs[0].plot(x,y)
axs[0].set_yticks([0,0.5, 1])
axs[0].set_xticks([0,0.5, 1])
axs[0].set_xlabel('FPR')
axs[0].set_ylabel('TPR')
axs[0].set_yticklabels(['0', '0.5', '1'])
axs[0].set_xticklabels(['0', '0.5', '1'])


axs[1].boxplot((total, filtered), showfliers=False)
axs[1].set_xticklabels(labels, fontsize=8)
axs[1].set_ylabel('No candidates')
fig.tight_layout(pad=3.0)
