import os
import sys
import multiprocessing
import datetime
def run_system_command(command):
    os.system(command)

# read from command line
args = sys.argv

dataset = args[1]
model = args[2]
control = args[3]

#dataset = 'Radar'
#model = 'classifier'
#control = '4_stack_50_10_search_0'

# get current time in str


# Higgs is too large, so we skip it
# datasets = [
#     ('CovType','classifier', '4_stack_50_10_search_0'),
#     ('MSD','linear', '4_stack_50_10_search_0'),
#     ('Gisette','classifier', '4_stack_50_10_search_0'),
#     ('Realsim','classifier', '4_stack_50_10_search_0'),
#     ('Epsilon','classifier', '4_stack_50_10_search_0'),
#     ('Letter', 'classifier', '4_stack_50_10_search_0'),
#     ('Radar', 'classifier', '4_stack_50_10_search_0'),
# ]

def run(pool, dataset, model, control):
    timestamp = datetime.datetime.now().strftime('%H-%M-%S')
    commands = [
        f'CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter corr --weight 0.0 > ./results/{dataset}_corr_0.0_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter corr --weight 0.3 > ./results/{dataset}_corr_0.3_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter corr --weight 0.6 > ./results/{dataset}_corr_0.6_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter corr --weight 1.0 > ./results/{dataset}_corr_1.0_{timestamp}.txt 2>&1',

        f'CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter imp --weight 0.1 > ./results/{dataset}_imp_0.1_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter imp --weight 0.3 > ./results/{dataset}_imp_0.3_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter imp --weight 0.6 > ./results/{dataset}_imp_0.6_{timestamp}.txt 2>&1',
        f'CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name {dataset} --model_name {model} --control_name {control} --init_seed 0 --splitter imp --weight 1.0 > ./results/{dataset}_imp_1.0_{timestamp}.txt 2>&1',
    ]

    # 将每条命令作为任务提交给进程池
    for command in commands:
        pool.apply_async(run_system_command, args=(command,))


if __name__ == "__main__":
    # 创建进程池并指定进程数量为4
    pool = multiprocessing.Pool(processes=8)


    
    # 关闭进程池，防止新的任务提交
    pool.close()

    # 等待所有任务完成
    pool.join()

    # 输出所有任务完成后的提示
    print('All system commands have been executed.')

