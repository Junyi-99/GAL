import multiprocessing
import os
import subprocess
import time
from queue import Empty


def process_wrapper(gpuid_queue, command, times):
    while True:
        try:
            gpu_idx = gpuid_queue.get(timeout=1)
        except Empty:
            break
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
        
        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} " + command + f'_repeat{times}.txt 2>&1'
        print("Running command: ", cmd)
        subprocess.call(cmd, shell=True)
        gpuid_queue.put(gpu_idx)
        break
    gpuid_queue.close()


def get_commands(dataset, model, control, dataseed):
    commands = [
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.0 --dataseed {dataseed} > ./results/{dataset}_corr_0.0',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.3 --dataseed {dataseed} > ./results/{dataset}_corr_0.3',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.6 --dataseed {dataseed} > ./results/{dataset}_corr_0.6',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 1.0 --dataseed {dataseed} > ./results/{dataset}_corr_1.0',

        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > ./results/{dataset}_imp_0.1',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.3 --dataseed {dataseed} > ./results/{dataset}_imp_0.3',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.6 --dataseed {dataseed} > ./results/{dataset}_imp_0.6',
        f'python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 1.0 --dataseed {dataseed} > ./results/{dataset}_imp_1.0',
    ]
    return commands

# datasets = [
#     ('CovType','classifier', '4_stack_50_10_search_0'), multicls
#     ('MSD','linear', '4_stack_50_10_search_0'), 
#     ('Gisette','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Realsim','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Epsilon','classifier', '4_stack_50_10_search_0'), 2cls
#     ('Letter', 'classifier', '4_stack_50_10_search_0'), 26cls
#     ('Radar', 'classifier', '4_stack_50_10_search_0'), 7cls
# ]

if __name__ == "__main__":
    num_gpus = 4
    gpuid_queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(processes=16)
    for i in range(num_gpus):
        gpuid_queue.put(i) # available gpu ids
        gpuid_queue.put(i)
        gpuid_queue.put(i) 
        gpuid_queue.put(i) # put 4次

    for ds in ['CovType', 'Gisette', 'Realsim', 'Epsilon', 'Radar', 'Letter']:
        for times in range(1, 5):
            commands = get_commands(ds, 'classifier', '4_stack_50_20_search_0', str(times))
            for cmd in commands:
                pool.apply_async(process_wrapper, (gpuid_queue, cmd, times))
    
    for ds in ['MSD']:
        for times in range(1, 5):
            commands = get_commands(ds, 'linear', '4_stack_50_20_search_0', str(times))
            for cmd in commands:
                pool.apply_async(process_wrapper, (gpuid_queue, cmd, times))
    print("Waiting for all subprocesses done...")
    pool.close()
    pool.join()
