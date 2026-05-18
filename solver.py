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


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(
            path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


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
        self.triplet_loss = TripletLossWithKL(
            B=self.batch_size, margin=1.0, device='cuda')
        
        self.best_f1 = None

    def build_model(self):
        self.model = AnomalyTransformer(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, data_path=self.data_path, 
                                        dataset=self.dataset, d_model=self.d_model, n_heads=self.n_heads, e_layers=self.e_layers, 
                                        d_ff=self.d_ff, dropout=self.dropout, attn_s_in=self.attn_s_in, max_b=self.batch_size, activation=self.activation, is_single_coords=self.is_single_coords)
        
        self.optimizer = torch.optim.Adam([p for n, p in self.model.named_parameters() if not n.endswith('.g')], lr=self.lr)
        self.optimizer_g = torch.optim.SGD([g for g in self.model.gs], lr=self.lr_g, momentum=0.9)

        self.optimizer_l1 = PGD([g for g in self.model.gs], proxs=[prox_operators.prox_l1_on_sigmoid], lr=1.0, alphas=[self.k2])
       
        if torch.cuda.is_available():
            self.model.cuda()

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []        
        for i, (input_data, _) in enumerate(vali_loader):
            input_ = input_data.float().to(self.device)
            output, series, prior, _, series_s, p_ss = self.model(input_)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()))+ torch.mean(my_kl_loss((prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(), series[u])))
                prior_loss += (torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()))+ torch.mean(my_kl_loss(series[u].detach(), (prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))))

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            rec_loss = self.criterion(output, input_)

            loss_1.append((rec_loss - self.k * series_loss).item())
            loss_2.append((rec_loss + self.k * prior_loss).item())
       
        return np.average(loss_1), np.mean(loss_2)
    
    def feature_smoothing(self, adj, X):
        rowsum = adj.sum(1)
        r_inv = rowsum.flatten()
        D = torch.diag(r_inv)
        L = D - adj

        r_inv = r_inv  + 1e-3
        r_inv = r_inv.pow(-1/2).flatten()
        r_inv[torch.isinf(r_inv)] = 0.
        r_mat_inv = torch.diag(r_inv)
        L = r_mat_inv @ L @ r_mat_inv

        loss_smooth_feat = torch.einsum('bnd,de,bne->b', X, L, X)
        return torch.mean(loss_smooth_feat)  

    def train(self):
        print("======================TRAIN MODE======================")
        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(
            patience=70, verbose=True, dataset_name=self.dataset)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input_ = input_data.float().to(self.device)

                output, series, prior, _, series_s, p_ss = self.model(input_)
                series_loss = 0.0
                prior_loss = 0.0
                series_s_loss = 0.0
                p_s_loss = 0.0

                for u in range(len(prior)):
                    series_loss += (torch.mean(my_kl_loss(series[u], (prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach()))
                                    + torch.mean(my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach(), series[u])))
                    prior_loss += (torch.mean(my_kl_loss((prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach()))
                                   + torch.mean(my_kl_loss(series[u].detach(), (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)))))
                    series_s_loss += (torch.mean(my_kl_loss_c(series_s[u], p_ss[u].detach()))+torch.mean(my_kl_loss_c(p_ss[u].detach(), series_s[u])))
                    p_s_loss += (torch.mean(my_kl_loss_c(series_s[u].detach(), p_ss[u]))+torch.mean(my_kl_loss_c(p_ss[u], series_s[u].detach())))
                    
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)
                series_s_loss = series_s_loss/len(series_s)
                p_s_loss = p_s_loss/len(p_ss)

                series_triplet_loss = self.triplet_loss(series)
                rec_loss = self.criterion(output, input_) 

                loss1 = rec_loss - self.k * series_loss + 1.0*series_triplet_loss - self.k1 * series_s_loss
                loss2 = rec_loss + self.k * prior_loss + self.k1*p_s_loss

                loss1_list.append((rec_loss - self.k * series_loss + 1.0*series_triplet_loss).item())

                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

                if (i + 1) % 5 == 0 and epoch > 1:
                    self.optimizer_g.zero_grad()
                    output, _, _, _, series_s, p_ss = self.model(input_)

                    prob_loss = 0.0
                    loss_smooth_feat = 0.0    
                    for u in range(len(series_s)):
                        prob_loss += (torch.mean(my_kl_loss_c(p_ss[u], series_s[u].detach()))+torch.mean(my_kl_loss_c(series_s[u].detach(), p_ss[u])))
                        loss_smooth_feat += self.feature_smoothing(torch.sigmoid(self.model.gs[u]), input_)
                        
                    prob_loss = prob_loss/len(series_s)
                    loss_smooth_feat = loss_smooth_feat/len(self.model.gs)
                    g_loss = self.criterion(output, input_)

                    loss1 = g_loss + self.k3*loss_smooth_feat + self.k1*prob_loss
                    loss1.backward()   

                    self.optimizer_g.step()
                    self.optimizer_l1.zero_grad()
                    self.optimizer_l1.step()

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            vali_loss1, vali_loss2 = self.vali(self.test_loader)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1))
            

            print(f'[LR-g] epoch={epoch+1} -> {self.optimizer_g.param_groups[0]["lr"]:.6g}')
            early_stopping(vali_loss1, vali_loss2, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

    def proc_batch_test(self, input_data, criterion):
        input_ = input_data.float().to(self.device)
        output, series, prior, _, series_s, p_ss = self.model(input_)
        loss = criterion(input_, output)

        series_loss = 0.0
        prior_loss = 0.0
        for u in range(len(prior)):
            series_loss += my_kl_loss(series[u], (prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)).detach())*self.t_t
            prior_loss += my_kl_loss((prior[u]/torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1, self.win_size)), series[u].detach())*self.t_t

        metric = torch.softmax((-series_loss - prior_loss), dim=-1)
        cri = torch.mean(loss, dim=-1)*metric              
        cri = cri.detach().cpu().numpy()  # [B,L]
        return cri

    def test(self):
        self.model, _, _ = load_checkpoint_flexibly(self.model, os.path.join(
            str(self.model_save_path), str(self.dataset) + '_checkpoint.pth'))
        self.model.eval()

        print("======================TEST MODE======================")
        criterion = nn.MSELoss(reduction='none')
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            cri = self.proc_batch_test(input_data, criterion)
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1) 
        train_energy = np.array(attens_energy)

        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            cri = self.proc_batch_test(input_data, criterion)
            attens_energy.append(cri)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anomaly_ratio)
        print("Threshold :", thresh)

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            cri = self.proc_batch_test(input_data, criterion)
            attens_energy.append(cri)
            test_labels.append(labels)
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        best_thresh, accuracy, precision, recall, f_score, fpr, ap = refine_threshold_with_metrics(
            test_energy, test_labels, thresh, window=0.15)
        print("Best threshold: ", best_thresh)
        print(
            "Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}, FPR : {:0.4f}, AP : {:0.4f}".format(
                accuracy, precision,
                recall, f_score, fpr, ap))
        return accuracy, precision, recall, f_score, fpr, ap