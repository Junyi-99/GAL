from torch.utils.data import Dataset
import pickle
import os
import torch
import sys
from os.path import dirname, abspath


def get_dataset(party_id:int, typ:str, value:str, train: str, dataseed: str):

    if typ =="corr":
        value = "beta" + value
    elif typ == "imp":
        value = "weight" + value
    else:
        assert False
    
    fname = ""
    if train == "train":
        fname = f"higgs_party4-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"
    else:
        fname = f"higgs_party4-{party_id}_{typ}_{value}_seed{dataseed}_test.pkl"
    
    dir_path = dirname(dirname(abspath(__file__)))
    sys.path.append(dir_path + "/vertibench")

    return pickle.load(open(f"/data/zhaomin/VertiBench/data/syn/higgs/{fname}", "rb"))

class Higgs(Dataset):
    # 二分类任务
    def __init__(self, root, split, typ: str, val:str, dataseed: str) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None, None, None, None]
        self.parties[0] = get_dataset(0, typ, val, split, dataseed) # 7 features
        self.parties[1] = get_dataset(1, typ, val, split, dataseed) # 7 features
        self.parties[2] = get_dataset(2, typ, val, split, dataseed) # 7 features
        self.parties[3] = get_dataset(3, typ, val, split, dataseed) # 7 features
        # total 28 features
        self.partitions = [
            self.parties[0].X.shape[1],
            self.parties[1].X.shape[1],
            self.parties[2].X.shape[1],
            self.parties[3].X.shape[1]
        ]
        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.X = torch.cat([torch.tensor(self.parties[0].X).clone().detach(),
                            torch.tensor(self.parties[1].X).clone().detach(),
                            torch.tensor(self.parties[2].X).clone().detach(),
                            torch.tensor(self.parties[3].X).clone().detach()
                        ], dim=1) # (8799999, 7)
        
        self.y[self.y == -1] = 0 # remap from [-1, 1] to [0, 1]
        self.target = self.y #  (8799999, 7) for training
        self.target_size = len(torch.unique(self.target))
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)
