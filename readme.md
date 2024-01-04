# RT-Tranformer
The RT-Tranformer combine the fingerprint and the molecular graph data and predict retention time as the output. It's architecture is showed as following:
![model](figs/model.svg)

# Motivation
Motivation:Liquid chromatography retention times prediction can assist in metabolite identification, which is a critical
task and challenge in non-targeted metabolomics. However, different chromatographic conditions may result in different
retention times for the same metabolite. Current retention time prediction methods lack sufficient scalability to transfer
from one specific chromatographic method to another


# Requirements
- Python 3.9
- torch
- rdkit-pypi
- torch-scatter
- torch-sparse 
- torch-cluster 
- torch_geometric
- scikit-learn
- tqdm
- jupyter
- notebook
- pandas
- networkx
- gradio

# Datasets
The SMRT dataset is collect from [this paper](https://doi.org/10.1038/s41467-019-13680-7)

All data and code are provided on [Google Drive]().
# Usage

## Validation and Test

Run test.py by `python ./test.py `


## Predict Rentention Time

You can follow the [jupyter notebook](./QuickStart.ipynb) to predict rentention time in your own data.

We provide easily accessible web pages and host them on the [huggingface](https://huggingface.co/spaces/junxue/RT-Transformer).

## Transfer Learn to Your Own Dataset

- Prepare your dataset as a csv file which has "InChI" and "RT" columns.
- Rename it as "data.csv" at the root directory.
- download the pre-trained model from [huggingface](https://huggingface.co/spaces/junxue/RT-Transformer/blob/main/best_state_dict.pth).
- Run transfer.py

You can also follow this [jupyter notebook](./) to fine-tuning the model.

## Retrain the Model
- Prepare your dataset as a csv file which has "InChI" and "RT" columns.
- Rename it as "data.csv" at the root directory.
- Run retrain.py

# Cite

If you make use of the code/experiment in your work, please cite our paper (Bibtex below).

    @article{xue2023rt,
    title={RT-Tranformer: Retention Time Prediction for Metabolite Annotation to Assist in Metabolite Identification},
    author={Jun Xue and Bingyi Wang and Hongchao Ji and Weihua Li },
    year={2023}
    }






    
