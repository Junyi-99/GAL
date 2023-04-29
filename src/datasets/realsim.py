from torch.utils.data import Dataset
import pickle
import os
import torch
import sys
from os.path import dirname, abspath


def get_dataset(party_id:int, typ:str, value:str, train: str):

    if typ =="corr":
        value = "beta" + value
    elif typ == "imp":
        value = "weight" + value
    else:
        assert False
    
    fname = ""
    if train == "train":
        fname = f"realsim_party4-{party_id}_{typ}_{value}_seed0_train.pkl"
    else:
        fname = f"realsim_party4-{party_id}_{typ}_{value}_seed0_test.pkl"
    
    dir_path = dirname(dirname(abspath(__file__)))
    sys.path.append(dir_path + "/vertibench")

    return pickle.load(open(f"/data/zhaomin/VertiBench/data/syn/realsim/{fname}", "rb"))

class Realsim(Dataset):
    # 二分类任务
    def __init__(self, root, split, typ: str, val:str) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None, None, None, None]
        self.parties[0] = get_dataset(0, typ, val, split)
        self.parties[1] = get_dataset(1, typ, val, split)
        self.parties[2] = get_dataset(2, typ, val, split)
        self.parties[3] = get_dataset(3, typ, val, split)
        
        self.key = list(torch.tensor(self.parties[0].key).clone().detach())
        self.y = torch.tensor(self.parties[0].y, dtype=torch.int64).clone().detach()
        self.X = torch.cat([torch.tensor(self.parties[0].X).clone().detach(), # (57847, 5241)
                            torch.tensor(self.parties[1].X).clone().detach(), # (57847, 5239)
                            torch.tensor(self.parties[2].X).clone().detach(), # (57847, 5239)
                            torch.tensor(self.parties[3].X).clone().detach()  # (57847, 5239)
                        ], dim=1)
        self.y[self.y == -1] = 0 # remap from [-1, 1] to [0, 1]
        self.target = self.y # (-1, 1)
        self.target_size = len(torch.unique(self.target)) # 个类
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)
