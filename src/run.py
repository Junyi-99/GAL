import os
import sys
import argparse 
import subprocess
import multiprocessing
from queue import Empty
import datetime, pytz

def process_wrapper(gpuid_queue, command, times):
    while True:
        try:
            gpu_idx = gpuid_queue.get(block=True, timeout=None)
        except Empty:
            continue
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_idx} " + command + f'_final_repeat{times}.txt 2>&1'
        print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Running command: ", cmd)
        subprocess.call(cmd, shell=True)
        gpuid_queue.put(gpu_idx)
        break
    gpuid_queue.close()


def get_commands(folder, lr, dataseed, dataset, model, control):
    commands = [
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.0 --dataseed {dataseed} > {folder}/corr_0.0.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.3 --dataseed {dataseed} > {folder}/corr_0.3.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 0.6 --dataseed {dataseed} > {folder}/corr_0.6.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter corr --weight 1.0 --dataseed {dataseed} > {folder}/corr_1.0.log',

        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 0.1 --dataseed {dataseed} > {folder}/imp_0.1.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 1.0 --dataseed {dataseed} > {folder}/imp_1.0.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 10.0 --dataseed {dataseed} > {folder}/imp_10.0.log',
        f'LR={lr} python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed {dataseed} --splitter imp --weight 100.0 --dataseed {dataseed} > {folder}/imp_100.0.log',
    ]
    return commands


def get_args():
    parser = argparse.ArgumentParser(description="GAL running script")  # 脚本描述
    parser.add_argument('-s', '--seeds', help="Random seed for dataset and GAL. You can specify multiple seeds to run. -s 0 1 2 3 4", nargs='+', type=int, required=True)
    parser.add_argument('-g', '--gpus', help="How many gpus to use. -g 1 2 3 4", nargs='+', type=int, required=True)
    parser.add_argument('-t', '--ntask', help="How many task you want to run on each gpu", type=int, required=True)
    parser.add_argument('-d', '--datasets', help="What datasets to run. -d MSD Gisette",nargs='+', type=str, required=True)
    parser.add_argument('-lr', '--learning-rates', help="Learning rate. You can specify multiple lr to run. -lr 0.1 0.01 0.001",nargs='+', type=str, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    gpus = args.gpus # number of gpus
    num_tasks = args.ntask
    
    gpuid_queue = multiprocessing.Manager().Queue()
    pool = multiprocessing.Pool(processes=len(gpus) * num_tasks)
    for i in gpus:
        for j in range(num_tasks):
            gpuid_queue.put(i) # available gpu ids

    for lr in args.learning_rates:
        for seed in args.seeds:
            for dataset in args.datasets:

                if dataset == 'MSD':
                    model = 'linear'
                    control = '4_stack_20_5_search_0'
                else:
                    model = 'classifier'
                    control = '4_stack_20_20_search_0'
                
                folder = f'./results/{dataset}_{model}_lr{lr}_{control}_seed{seed}'
                os.system(f'mkdir -p {folder}')
                commands = get_commands(folder, lr, seed, dataset, model, control)
                for cmd in commands:
                    pool.apply_async(process_wrapper, (gpuid_queue, cmd, seed))
    
    print(datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S"), "Waiting for all subprocesses done...")
    pool.close()
    pool.join()
