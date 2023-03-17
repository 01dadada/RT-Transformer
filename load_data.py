from typing import Optional, Callable, Union, List, Tuple
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
import numpy as np
from collections import defaultdict
from torch_geometric.data import Data, in_memory_dataset, Dataset, InMemoryDataset
from torch_geometric.loader import DataLoader
import numpy as np
import os
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_networkx, to_dense_adj
import networkx as nx
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import random
import os.path as osp
import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from sklearn.preprocessing import OneHotEncoder
from rdkit.Chem.Scaffolds import MurckoScaffold
# import warnings
#
# warnings.filterwarnings("ignore")

CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE,
    BT.DOUBLE,
    BT.TRIPLE,
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]


class SMRTDatasetScaffold(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return './SMRT_dataset.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('SMRT_dataset.csv', sep=';')
        y = res['rt']
        inchi_list = res['inchi']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, inchi in enumerate(inchi_list):
            try:
                # mol = Chem.MolFromInchi(inchi,sanitize=True,removeHs=False)
                mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []
                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))

                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)

                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14, ], dim=-1)

                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                scaffold = ''
                try:
                    scaffold = MurckoScaffoldSmiles(mol=mol)
                except:
                    print("No scaffold")
                data = Data(x=x, y=torch.tensor(res['rt'][index], dtype=torch.float32),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            fingerprint=fingerprint,
                            # inchi=res['inchi'][index],
                            # formula=Chem.rdMolDescriptors.CalcMolFormula(mol),
                            MurckoScaffold=str(scaffold))
                data_list.append(data)
            except Exception as re:
                print('ERROR,此分子无法转换')
                print(re)
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SMRTDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return './SMRT_dataset.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('SMRT_dataset.csv', sep=';')
        y = res['rt']
        inchi_list = res['inchi']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, inchi in enumerate(inchi_list):
            try:
                mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []
                atom_is_aromatic = []


                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)

                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x,
                            y=torch.tensor(res['rt'][index], dtype=torch.float32),
                            edge_index=edge_index,
                            edge_attr=edge_attr,
                            fingerprint=fingerprint,
                            inchi=res['inchi'][index],
                            formula=Chem.rdMolDescriptors.CalcMolFormula(mol))
                data_list.append(data)
            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()



class SMRTDatasetRetained(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'retain_data.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('retain_data.csv')
        y = res['rt']
        inchi_list = res['inchi']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, inchi in enumerate(inchi_list):
            try:
                mol = Chem.MolFromInchi(inchi, sanitize=False, removeHs=False)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,
                               # x15
                               ], dim=-1)

                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(res['rt'][index], dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint, inchi=res['inchi'][index],
                            formula=Chem.rdMolDescriptors.CalcMolFormula(mol))
                # data = Data(x=x, y=torch.tensor(res['rt'][index], dtype=torch.float32), edge_index=edge_index,
                #             edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)
            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

class RikenDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'smiles_retention_Riken_Retip.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('smiles_retention_Riken_Retip.csv')
        y = res['rt']
        smiles_list = res['smiles_character']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)

            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

class MassBank1Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'smiles_retention_MassBank1.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('smiles_retention_MassBank1.csv')
        y = res['retention_time']
        smiles_list = res['smiles']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)

            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

class MetabobaseDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'smiles_retention_Metabobase.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('smiles_retention_Metabobase.csv')
        y = res['retention_time']
        smiles_list = res['smiles']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)

            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()

class Retntion_Life_Dataset_Old(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMILES_Retention_Life_Old.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('SMILES_Retention_Life_Old.csv')
        y = res['recorded_rt']
        smiles_list = res['SMILES']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)

            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()


class Retntion_Life_Dataset_New(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'SMILES_Retention_Life_New.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('SMILES_Retention_Life_New.csv')
        y = res['retention_time']
        smiles_list = res['SMILES']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0
        x_list = []
        edge_index_list = []
        edge_attr_list = []

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)

            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()


class HilicDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        return 'smiles_retention_Hilic_Retip.csv'

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        res = pd.read_csv('smiles_retention_Hilic_Retip.csv',sep=',')
        y = res['Experimental Retention Time']
        smiles_list = res['smiles_character']

        hybridization_list = ['OTHER', 'S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
        hybridization_encoder = OneHotEncoder()
        hybridization_encoder.fit(torch.range(0, len(hybridization_list) - 1).unsqueeze(-1))

        atom_list = ['H', 'C', 'O', 'S', 'N', 'P', 'F', 'Cl', 'Br', 'I', 'Si']
        atom_encoder = OneHotEncoder()
        atom_encoder.fit(torch.range(0, len(atom_list) - 1).unsqueeze(-1))

        chirarity_encoder = OneHotEncoder()
        chirarity_encoder.fit(torch.range(0, len(CHIRALITY_LIST) - 1).unsqueeze(-1))

        data_list = []
        i = 0

        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                mol = Chem.AddHs(mol)

                weights = []
                type_idx = []
                chirality_idx = []
                atomic_number = []
                degrees = []
                total_degrees = []
                formal_charges = []
                hybridization_types = []
                explicit_valences = []
                implicit_valences = []
                total_valences = []
                atom_map_nums = []
                isotopes = []
                radical_electrons = []
                inrings = []

                atom_is_aromatic = []

                for atom in mol.GetAtoms():
                    atom_is_aromatic.append(atom.GetIsAromatic())
                    type_idx.append(atom_list.index(atom.GetSymbol()))
                    chirality_idx.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
                    atomic_number.append(atom.GetAtomicNum())
                    degrees.append(atom.GetDegree())
                    weights.append(atom.GetMass())
                    total_degrees.append(atom.GetTotalDegree())
                    formal_charges.append(atom.GetFormalCharge())
                    hybridization_types.append(hybridization_list.index(str(atom.GetHybridization())))
                    explicit_valences.append(atom.GetExplicitValence())
                    implicit_valences.append(atom.GetImplicitValence())
                    total_valences.append(atom.GetTotalValence())
                    atom_map_nums.append(atom.GetAtomMapNum())
                    isotopes.append(atom.GetIsotope())
                    radical_electrons.append(atom.GetNumRadicalElectrons())
                    inrings.append(int(atom.IsInRing()))


                x1 = torch.tensor(type_idx, dtype=torch.float32).view(-1, 1)
                x2 = torch.tensor(chirality_idx, dtype=torch.float32).view(-1, 1)
                x3 = torch.tensor(weights, dtype=torch.float32).view(-1, 1)
                x4 = torch.tensor(degrees, dtype=torch.float32).view(-1, 1)
                x5 = torch.tensor(total_degrees, dtype=torch.float32).view(-1, 1)
                x6 = torch.tensor(formal_charges, dtype=torch.float32).view(-1, 1)
                x7 = torch.tensor(hybridization_types, dtype=torch.float32).view(-1, 1)
                x8 = torch.tensor(explicit_valences, dtype=torch.float32).view(-1, 1)
                x9 = torch.tensor(implicit_valences, dtype=torch.float32).view(-1, 1)
                x10 = torch.tensor(total_valences, dtype=torch.float32).view(-1, 1)
                x11 = torch.tensor(atom_map_nums, dtype=torch.float32).view(-1, 1)
                x12 = torch.tensor(isotopes, dtype=torch.float32).view(-1, 1)
                x13 = torch.tensor(radical_electrons, dtype=torch.float32).view(-1, 1)
                x14 = torch.tensor(inrings, dtype=torch.float32).view(-1, 1)
                # x15 =  torch.tensor(atom_is_aromatic, dtype=torch.float32).view(-1, 1)
            
                # x = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14]

                x = torch.cat([torch.tensor(atom_encoder.transform(x1).toarray(), dtype=torch.float32),
                               torch.tensor(chirarity_encoder.transform(x2).toarray(), dtype=torch.float32),
                               x3,
                               x4,
                               x5,
                               x6,
                               torch.tensor(hybridization_encoder.transform(x7).toarray(), dtype=torch.float32),
                               x8,
                               x9,
                               x10,
                               x11,
                               x12,
                               x13,
                               x14,], dim=-1)


                row, col, edge_feat = [], [], []
                for bond in mol.GetBonds():
                    start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                    row += [start, end]
                    col += [end, start]
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                    edge_feat.append([
                        BOND_LIST.index(bond.GetBondType()),
                        BONDDIR_LIST.index(bond.GetBondDir()),
                        float(int(bond.IsInRing())),
                        float(int(bond.GetIsAromatic())),
                        float(int(bond.GetIsConjugated()))
                    ])
                edge_index = torch.tensor([row, col], dtype=torch.long)
                edge_attr = torch.tensor(np.array(edge_feat), dtype=torch.float32)
                fingerprint = torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, 2), dtype=torch.float32)
                data = Data(x=x, y=torch.tensor(y[index]*60, dtype=torch.float32), edge_index=edge_index,
                            edge_attr=edge_attr, fingerprint=fingerprint)
                # x_list.append(x)
                # edge_index_list.append(edge_index)
                # edge_attr_list.append(edge_attr)
                data_list.append(data)
                print(index)
            except:
                print(index)
                pass
            i = i + 1
        print(data_list.__len__())
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_filter is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def plot_graph(self, idx):
        data = self.get(idx)
        g = to_networkx(data, to_undirected=True)
        color_map = nx.get_node_attributes(g, "label")
        values = [color_map.get(node) for node in g.nodes()]
        pos = nx.spring_layout(g, seed=3113794652)
        plt.figure(figsize=(8, 8))
        nx.draw(g, pos, cmap=plt.get_cmap('viridis'), node_color=values, node_size=80,
                linewidths=6)  # with_labels=True
        plt.show()


if __name__ == '__main__':

    print('Loading ...')
    dataset = SMRTDataset('./SMRT')
    print('Number of graphs in dataset: ', len(dataset))
    print(dataset.get(0).fingerprint)
    dataset.plot_graph(len(dataset)-1)
