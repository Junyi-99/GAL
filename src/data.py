import torch
import numpy as np
from config import cfg
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


def fetch_dataset(data_name, args, verbose=True):
    import datasets
    dataset = {}
    if verbose:
        print('fetching data {}...'.format(data_name))
    root = './data/{}'.format(data_name)
    if data_name in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in ['MNIST', 'CIFAR10']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
    elif data_name in ['ModelNet40', 'ShapeNet55']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\', '
                                'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\', '
                               'transform=datasets.Compose([transforms.ToTensor()]))'.format(data_name))
    elif data_name in ['MIMICL', 'MIMICM']:
        dataset['train'] = eval('datasets.{}(root=root, split=\'train\')'.format(data_name))
        dataset['test'] = eval('datasets.{}(root=root, split=\'test\')'.format(data_name))
    elif data_name in ['MSD', 'CovType', 'Higgs', 'Gisette', 'Letter', 'Radar', 'Epsilon', 'Realsim']:
        dataset['train'] = eval(f"datasets.{data_name}(root=root, split='train', typ='{args['splitter']}', val='{args['weight']}')")
        dataset['test'] = eval(f"datasets.{data_name}(root=root, split='test', typ='{args['splitter']}', val='{args['weight']}')")
    else:
        raise ValueError('Not valid dataset name')
    if verbose:
        print('data ready')
    return dataset


def input_collate(batch):
    if isinstance(batch[0], dict):
        output = {key: [] for key in batch[0].keys()}
        for b in batch:
            for key in b:
                output[key].append(b[key])
        return output
    else:
        return default_collate(batch)


def make_data_loader(dataset, tag, shuffle=None):
    data_loader = {}
    for k in dataset:
        _shuffle = cfg[tag]['shuffle'][k] if shuffle is None else shuffle[k]
        data_loader[k] = DataLoader(dataset=dataset[k], shuffle=_shuffle, batch_size=cfg[tag]['batch_size'][k],
                                    pin_memory=cfg['pin_memory'], num_workers=cfg['num_workers'],
                                    collate_fn=input_collate,
                                    worker_init_fn=np.random.seed(cfg['seed']))
    return data_loader


def split_dataset(num_users):
    if cfg['data_name'] in ['Blob', 'Iris', 'Diabetes', 'BostonHousing', 'Wine', 'BreastCancer', 'QSAR', 'ModelNet40',
                            'ShapeNet55']:
        num_features = cfg['data_shape'][-1]
        feature_split = list(torch.randperm(num_features).split(num_features // num_users))
        feature_split = feature_split[:num_users - 1] + [torch.cat(feature_split[num_users - 1:])]
    elif cfg['data_name'] in ['MIMICL', 'MIMICM']:
        if cfg['num_users'] == 1:
            feature_split = [list(range(22))]
        elif cfg['num_users'] == 4:
            feature_split = [None for _ in range(4)]
            feature_split[0] = list(range(16))
            feature_split[1] = list(range(16, 19))
            feature_split[2] = list(range(19, 21))
            feature_split[3] = [21]
        else:
            raise ValueError('Not valid num users')
    elif cfg['data_name'] in ['MNIST', 'CIFAR10']:
        num_features = np.prod(cfg['data_shape']).item()
        idx = torch.arange(num_features).view(*cfg['data_shape'])
        power = np.log2(num_users)
        n_h, n_w = int(2 ** (power // 2)), int(2 ** (power - power // 2))
        feature_split = idx.view(cfg['data_shape'][0], n_h, cfg['data_shape'][1] // n_h, n_w,
                                 cfg['data_shape'][2] // n_w).permute(1, 3, 0, 2, 4).reshape(
            -1, cfg['data_shape'][0], cfg['data_shape'][1] // n_h, cfg['data_shape'][2] // n_w)
        feature_split = list(feature_split.reshape(feature_split.size(0), -1))
    elif cfg['data_name'] in ['MSD']:
        # 4个party分别有24，22，22，22个 feature
        feature_split = [
            torch.arange(0, 24, dtype=torch.int),
            torch.arange(24, 46, dtype=torch.int),             
            torch.arange(46, 68, dtype=torch.int),
            torch.arange(68, 90, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['CovType']:
        feature_split = [
            torch.arange(0, 15, dtype=torch.int),
            torch.arange(15, 28, dtype=torch.int),
            torch.arange(28, 41, dtype=torch.int),
            torch.arange(41, 54, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Higgs']:
        feature_split = [
            torch.arange(0, 7, dtype=torch.int),
            torch.arange(7, 14, dtype=torch.int),
            torch.arange(14, 21, dtype=torch.int),
            torch.arange(21, 28, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Gisette']:
        feature_split = [
            torch.arange(0, 1250, dtype=torch.int),
            torch.arange(1250, 2500, dtype=torch.int),
            torch.arange(2500, 3750, dtype=torch.int),
            torch.arange(3750, 5000, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Realsim']:
        feature_split = [
            torch.arange(0, 5241, dtype=torch.int),
            torch.arange(5241, 10480, dtype=torch.int),
            torch.arange(10480, 15719, dtype=torch.int),
            torch.arange(15719, 20958, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Epsilon']: 
        feature_split = [
            torch.arange(0, 500, dtype=torch.int),
            torch.arange(500, 1000, dtype=torch.int),
            torch.arange(1000, 1500, dtype=torch.int),
            torch.arange(1500, 2000, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Letter']:
        feature_split = [
            torch.arange(0, 4, dtype=torch.int),
            torch.arange(4, 8, dtype=torch.int),
            torch.arange(8, 12, dtype=torch.int),
            torch.arange(12, 16, dtype=torch.int)
        ]
    elif cfg['data_name'] in ['Radar']:
        feature_split = [
            torch.arange(0, 45, dtype=torch.int),
            torch.arange(45, 88, dtype=torch.int),
            torch.arange(88, 131, dtype=torch.int),
            torch.arange(131, 174, dtype=torch.int)
        ]
    else:
        raise ValueError('Not valid data name')
    return feature_split
