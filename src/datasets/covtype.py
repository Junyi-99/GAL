from torch.utils.data import Dataset
import pickle
import os
import torch
import sys
from os.path import dirname, abspath

# 一共输入 90 feature, 4 party，370972 个样本
# 一共输出 1 feature，4 party，370972 个样本
# party0 有 24 feature
# party1 有 22 feature
# party2 有 22 feature
# party3 有 22 feature

# 获取zhaomin的数据集
def get_dataset(party_id:int, typ:str, value:str, train: str, dataseed: str):
    # dataseed: 0,1,2,3,4
    if typ =="corr":
        value = "beta" + value
    elif typ == "imp":
        value = "weight" + value
    else:
        assert False
    
    fname = ""
    if train == "train":
        fname = f"covtype_party4-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"
    else:
        fname = f"covtype_party4-{party_id}_{typ}_{value}_seed{dataseed}_test.pkl"
    
    dir_path = dirname(dirname(abspath(__file__)))
    sys.path.append(dir_path + "/vertibench")

    msd = pickle.load(open(f"/data/zhaomin/VertiBench/data/syn/covtype/{fname}", "rb"))
    return msd

class CovType(Dataset):
    def __init__(self, root, split, typ: str, val:str, dataseed: str) -> None:
        super().__init__()
        self.root = os.path.expanduser(root)
        self.split = split
        # split: train, test
        # typ: corr, imp
        # val: 0.1, 0.3, 0.5, 0.7, 0.9


        self.parties = [None, None, None, None]
        self.parties[0] = get_dataset(0, typ, val, split, dataseed) # 15 features
        self.parties[1] = get_dataset(1, typ, val, split, dataseed) # 13 features
        self.parties[2] = get_dataset(2, typ, val, split, dataseed) # 13 features
        self.parties[3] = get_dataset(3, typ, val, split, dataseed) # 13 features
        # total 54 features
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
                        ], dim=1) # (464809, 54)

        self.target = self.y # (464809, 1) for training, (116203, 1) for testing
        self.target_size = len(torch.unique(self.target))
        
    def __getitem__(self, index):
        input = {'id': self.key[index], 'data': self.X[index], 'target': self.y[index]}
        return input

    def __len__(self) -> int:
        return len(self.key)
