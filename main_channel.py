import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from solver_channel import Solver
from utils.utils import *
from torch.backends import cudnn
import time
import sys

def apply_overrides_from_argv(cfg_train, cfg_test):
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        key = args[i]
        if not key.startswith("--"):
            continue
        key = key[2:]
        val = args[i + 1]
        if key in cfg_train:
            cfg_train[key] = type(cfg_train[key])(val)
        if key in cfg_test:
            cfg_test[key] = type(cfg_test[key])(val)

def str2bool(v):
    return v.lower() in ('true')

def main(config_train, config_test):
    cudnn.benchmark = True
    if (not os.path.exists(config_train['model_save_path'])):
        mkdir(config_train['model_save_path'])
    solver_train = Solver(config_train)
    solver_test = Solver(config_test)
    
    t1 = time.perf_counter()
    solver_test.test()
    t_test = time.perf_counter() - t1
    print(f"[TIMER] TEST_SEC={t_test:.2f}")

def getRoc(config_test):
    if (not os.path.exists(config_train['model_save_path'])):
        mkdir(config_train['model_save_path'])
    solver_test = Solver(config_test)
    t1 = time.perf_counter()
    accuracy, precision, recall, f_score, fpr, ap = solver_test.test()
    t_test = time.perf_counter() - t1
    print(f"[TIMER] TEST_SEC={t_test:.2f}")
    print(f"[TIMER] TOTAL_SEC={(t_train if 't_train' in locals() else 0) + t_test:.2f}")
    return accuracy, precision, recall, f_score, fpr


if __name__ == '__main__':

    config_train = {'lr': 1e-4, 'num_epochs': 3, 'k': 3, 'win_size': 100, 'input_c': 38, 'output_c': 38, 'batch_size': 32,
                    'dataset': 'SMD+', 'mode': 'train', 'd_model': 512, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512,
                    'dropout': 0.0, 'attn_s_in': True, 'activation': 'gelu',
                    'data_path': './dataset/SMD+', 'is_single_coords': True,
                    'model_save_path': './checkpoints',
                    'anomaly_ratio': 0.5, 'k1': 0.09, 'k2': 0.8, 'k3': 0.004, 'k4': 1.0, 'lr_g': 0.01, 't_t': 1000.0, 't_s': 0.1}
    config_test = {'lr': 1e-4, 'num_epochs': 3, 'k': 3, 'win_size': 100, 'input_c': 38, 'output_c': 38, 'batch_size': 32,
                   'dataset': 'SMD+', 'mode': 'test', 'd_model': 512, 'n_heads': 8, 'e_layers': 3, 'd_ff': 512,
                   'dropout': 0.0, 'attn_s_in': True, 'activation': 'gelu',
                   'data_path': './dataset/SMD+', 'is_single_coords': True,
                   'model_save_path': './checkpoints',
                   'anomaly_ratio': 0.5, 'k1': 0.09, 'k2': 0.8, 'k3': 0.004, 'k4': 1.0, 'lr_g': 0.01, 't_t': 1000.0, 't_s': 0.1}
    
    apply_overrides_from_argv(config_train, config_test)

    main(config_train, config_test)