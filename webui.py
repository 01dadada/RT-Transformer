import time
import gradio as gr
from model import MyNet
from transferDataset import get_data_list
from rdkit import Chem
import torch
import os
import pandas as pd
import hashlib
from torch_geometric.loader import DataLoader


model = MyNet(emb_dim=512, feat_dim=512)
state = torch.load('./best_state_download_dict.pth')
model.load_state_dict(state)
model.eval()

def get_rt_from_mol(mol):
    data_list = get_data_list([mol])
    loader = DataLoader(data_list,batch_size=1)
    for batch in loader:
        break
    return model(batch).item()

def pred_file_btyes(file_bytes,progress=gr.Progress()):
    progress(0,desc='Starting')
    file_name = os.path.join(
        './save_df/',
        (hashlib.md5(str(file_bytes).encode('utf-8')).hexdigest()+'.csv')
        )
    if os.path.exists(file_name):
        print('该文件已经存在')
        return file_name
    with open('temp.sdf','bw') as f:
        f.write(file_bytes)
    sup = Chem.SDMolSupplier('temp.sdf')
    df = pd.DataFrame(columns=['InChI','Predicted RT'])
    for mol in progress.tqdm(sup):
        try:
            inchi = Chem.MolToInchi(mol)
            rt = get_rt_from_mol(mol)
            df.loc[len(df)] = [inchi,rt]
        except:
            pass
    
    df.to_csv(file_name)
    return file_name

demo = gr.Interface(
    pred_file_btyes, 
    gr.File(type='binary'), 
    gr.File(type='filepath'),
    title='RT-Transformer Rentention Time Predictor',
    description='Input SDF Molecule File,Predicted RT output with a CSV File',
    )


if __name__ == "__main__":
    demo.launch()
