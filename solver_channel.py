import torch
import torch.nn as nn
import numpy as np
import os
import time
from utils.utils import *
from utils.metrics import *
from utils.pgd import PGD, prox_operators
from model.AnomalyTransformer import AnomalyTransformer
from model.constraints import *
from data_factory.data_loader import get_loader_segment

from datetime import datetime

SNAPSHOT_ROOT = "./nan_snapshots"
os.makedirs(SNAPSHOT_ROOT, exist_ok=True)


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def my_kl_loss_c(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.sum(res, dim=-1)

def load_checkpoint_flexibly(model, path, verbose=False):
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    matched, skipped = [], []

    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            new_state_dict[k] = v
            matched.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(new_state_dict, strict=False)
    return model, matched, skipped


class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.train_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(
            self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)
        self.build_model()
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        
        self.best_f1 = None

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, data_path=self.data_path, 
                                        dataset=self.dataset, d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers, 
                                        d_ff=self.d_ff, dropout=self.dropout, attn_s_in=self.attn_s_in, max_b=self.batch_size, activation=self.activation, is_single_coords=self.is_single_coords)
               
        if torch.cuda.is_available():
            self.model.cuda()
  
    def proc_batch_test(self, input_data, criterion, global_mean=None, global_std=None):
        input_ = input_data.float().to(self.device)
        output, series, prior, _, series_s, p_ss = self.model(input_)        
        rec_loss = criterion(input_, output) 

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += my_kl_loss(series[u], (prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()) * self.t_t
            prior_loss += my_kl_loss((prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()) * self.t_t
        
        metric = torch.softmax((-series_loss - prior_loss), dim=-1)

        series_s_loss = 0.0
        prob_loss = 0.0
        for u in range(len(series_s)):
            series_s_loss += my_kl_loss_c(series_s[u], p_ss[u].detach())
            prob_loss += my_kl_loss_c(p_ss[u], series_s[u].detach())
            
        raw_metric_s = -series_s_loss - prob_loss 
        if global_mean is not None and global_std is not None:
            z_metric_s = (raw_metric_s - global_mean) / global_std
        else:
            mean_s = raw_metric_s.mean(dim=-1, keepdim=True)
            std_s = raw_metric_s.std(dim=-1, keepdim=True) + 1e-8
            z_metric_s = (raw_metric_s - mean_s) / std_s
            
        metric_s = torch.sigmoid(z_metric_s * self.t_s)
        cri = rec_loss * torch.unsqueeze(metric, dim=-1) * torch.unsqueeze(metric_s, dim=1)
        
        return cri.detach().cpu().numpy()

    def _get_global_spatial_stats(self):
            print("Calculating Global Historical Statistics on Train Set...")
            self.model.eval()
            raw_metrics_list = []
            with torch.no_grad():
                for i, (input_data, _) in enumerate(self.train_loader):
                    input_ = input_data.float().to(self.device)
                    _, _, _, _, series_s, p_ss = self.model(input_)
                    
                    series_s_loss = 0.0
                    prob_loss = 0.0
                    for u in range(len(series_s)):
                        series_s_loss += my_kl_loss_c(series_s[u], p_ss[u].detach())
                        prob_loss += my_kl_loss_c(p_ss[u], series_s[u].detach())
                    
                    raw_metric_s = -series_s_loss - prob_loss 
                    raw_metrics_list.append(raw_metric_s.cpu())
                    
            all_raw_metrics = torch.cat(raw_metrics_list, dim=0)
            global_mean = all_raw_metrics.mean(dim=0, keepdim=True).to(self.device)
            global_std = all_raw_metrics.std(dim=0, keepdim=True).to(self.device) + 1e-8
            print("Global Statistics Calculated!")
            return global_mean, global_std

    def test(self):
        self.model, _, _ = load_checkpoint_flexibly(self.model, os.path.join(
            str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'))
        self.model.eval()

        print("======================TEST MODE (Strict Channel-wise)======================")
        criterion = nn.MSELoss(reduction='none')
        global_mean, global_std = self._get_global_spatial_stats()

        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            cri = self.proc_batch_test(input_data, criterion, global_mean=global_mean, global_std=global_std)
            attens_energy.append(cri)
        
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.output_c)
        train_energy = np.array(attens_energy)

        attens_energy = []
        test_labels = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            cri = self.proc_batch_test(input_data, criterion, global_mean=global_mean, global_std=global_std)
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1, self.output_c)
        test_energy = np.array(attens_energy)
        test_labels_raw = np.concatenate(test_labels, axis=0)
        
        if test_labels_raw.ndim == 3: 
             test_labels = test_labels_raw.reshape(-1, self.output_c)
        elif test_labels_raw.ndim == 2:              
             if test_labels_raw.shape[1] == self.output_c:
                 test_labels = test_labels_raw 
             else:
                 raise ValueError(f"Loaded labels shape {test_labels_raw.shape} mismatch. Expected (N, {self.output_c}). Check if SMD_Synthetic_test_label_ND.npy is loaded.")
        else:
             raise ValueError("Labels are 1D. Strict Channel-wise evaluation requires N*D labels.")

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy.ravel(), 100 - self.anomaly_ratio)
        print("Initial Threshold (based on Train+Test distribution):", thresh)

        print(f"Starting threshold refinement (Data shape: {test_energy.shape})...")
            
        t_start = time.perf_counter()            
        best_thresh, accuracy, precision, recall, f_score, fpr, ap = refine_threshold_channel_wise(test_energy, test_labels, thresh, window=0.15)
            
        t_end = time.perf_counter()
        elapsed = t_end - t_start
        print(f"[Timer] refine_threshold_channel_wise took: {elapsed:.4f} seconds")
        
        print("Best threshold: ", best_thresh)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, FPR : {:0.4f}, AP : {:0.4f}".format(
                accuracy, precision,
                recall, f_score, fpr, ap))
        
        return accuracy, precision, recall, f_score, fpr, ap
