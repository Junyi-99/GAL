import pickle


# 获取zhaomin的数据集
def get_dataset(dataset_name: str, party_id:int, typ:str, value:str, train: str, dataseed: str, num_clients: int):
    # dataseed: 0,1,2,3,4
    if typ =="corr":
        value = "beta" + value
    elif typ == "imp": # 0.1，1.0，10.0，100.0
        value = "weight" + value
    else:
        assert False
    
    fname = ""
    if train == "train":
        fname = f"{dataset_name}_party{num_clients}-{party_id}_{typ}_{value}_seed{dataseed}_train.pkl"
    else:
        fname = f"{dataset_name}_party{num_clients}-{party_id}_{typ}_{value}_seed{dataseed}_test.pkl"

    msd = pickle.load(open(f"/data/zhaomin/VertiBench/data/syn/{dataset_name}/{fname}", "rb"))
    return msd