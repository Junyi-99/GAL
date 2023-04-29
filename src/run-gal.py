import os
import sys
import multiprocessing
import datetime
import time
def run_system_command(command):
    
    os.system(command)
    time.sleep(3)

#dataset = "Radar"
#model = "classifier"
#control = "4_stack_50_10_search_0"

# get current time in str
timestamp = datetime.datetime.now().strftime("%H-%M-%S")

commands = [
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.0 > Realsim_corr_0.0_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.3 > Realsim_corr_0.3_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.6 > Realsim_corr_0.6_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 1.0 > Realsim_corr_1.0_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.1 > Realsim_imp_0.1_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.3 > Realsim_imp_0.3_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.6 > Realsim_imp_0.6_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Realsim --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 1.0 > Realsim_imp_1.0_06-28-19.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.0 > Epsilon_corr_0.0_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.3 > Epsilon_corr_0.3_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.6 > Epsilon_corr_0.6_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 1.0 > Epsilon_corr_1.0_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.1 > Epsilon_imp_0.1_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.3 > Epsilon_imp_0.3_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.6 > Epsilon_imp_0.6_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Epsilon --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 1.0 > Epsilon_imp_1.0_06-28-29.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.0 > Gisette_corr_0.0_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.3 > Gisette_corr_0.3_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.6 > Gisette_corr_0.6_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 1.0 > Gisette_corr_1.0_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.1 > Gisette_imp_0.1_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.3 > Gisette_imp_0.3_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.6 > Gisette_imp_0.6_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Gisette --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 1.0 > Gisette_imp_1.0_06-32-24.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.0 > MSD_corr_0.0_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.3 > MSD_corr_0.3_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.6 > MSD_corr_0.6_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 1.0 > MSD_corr_1.0_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.1 > MSD_imp_0.1_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.3 > MSD_imp_0.3_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.6 > MSD_imp_0.6_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name MSD --model_name linear --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 1.0 > MSD_imp_1.0_06-41-48.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.0 > Higgs_corr_0.0_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.3 > Higgs_corr_0.3_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 0.6 > Higgs_corr_0.6_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter corr --weight 1.0 > Higgs_corr_1.0_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=0 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.1 > Higgs_imp_0.1_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=1 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.3 > Higgs_imp_0.3_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=2 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 0.6 > Higgs_imp_0.6_06-42-09.txt 2>&1",
"CUDA_VISIBLE_DEVICES=3 python train_model_assist.py --data_name Higgs --model_name binclassifier --control_name 4_stack_10_10_search_0 --init_seed 0 --splitter imp --weight 1.0 > Higgs_imp_1.0_06-42-09.txt 2>&1",
]

if __name__ == "__main__":
    # 创建进程池并指定进程数量为4
    pool = multiprocessing.Pool(processes=8)

    # 将每条命令作为任务提交给进程池
    for command in commands:
        pool.apply_async(run_system_command, args=(command,))

    # 关闭进程池，防止新的任务提交
    pool.close()

    # 等待所有任务完成
    pool.join()

    # 输出所有任务完成后的提示
    print('All system commands have been executed.')
