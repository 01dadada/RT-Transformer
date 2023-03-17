

[//]: # (# 文件说明)

[//]: # (    load_data.py SMRT数据集dataset)

[//]: # (    model.py                        模型核心文件)

[//]: # (    train.py                        核心文件，生成预训练模型)

[//]: # (    test.py                         测试预训练模型)

[//]: # (    test_save.py                    测试预训练模型并保存结果)

[//]: # (    transferFinal.py                对5个迁移数据集进行迁移)

[//]: # (    test_transfer_all.py            对所有迁移数据集进行迁移)

[//]: # (    tranferDataset.py               迁移数据集)

[//]: # (    windows_transfer.py             对5个迁移数据集进行迁移&#40;windows&#41;)

[//]: # (    predict_selections.py           对从SMRT数据集的测试集选出的100的分子，搜索出的所有结构进行保留时间预测)

[//]: # (    predict_for_selections.py       对从SMRT数据集的测试集选出的100的分子进行保留时间预测)


# environment
    cuda=11.3.1
    python=3.9

    pip3 install torch==1.11.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

    pip install torch-scatter==2.0.9 torch-sparse==0.6.14 torch-cluster==1.6.0  torch-geometric==2.0.4 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
    
    pip install matplotlib networkx torchmetrics seaborn 
    
    pip install rdkit==2022.9.1

    


    
