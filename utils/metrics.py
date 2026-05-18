from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score,confusion_matrix,average_precision_score
from scipy.optimize import minimize_scalar
import numpy as np

def adjust_anomaly_channel_wise(pred, gt):

    pred = pred.astype(int)
    gt = gt.astype(int)

    for col in range(pred.shape[1]):
        this_gt = gt[:, col]
        
        if np.sum(this_gt) > 0:
            pred[:, col] = adjust_anomaly_p(pred[:, col], this_gt)
            
    return pred

def adjust_anomaly(pred, gt):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1

def adjust_anomaly_p(pred, gt):
    pred = pred.astype(int)
    gt = gt.astype(int)

    if np.sum(pred) == 0:
        return pred

    padded_gt = np.pad(gt, (1, 1), 'constant', constant_values=0)
    diffs = np.diff(padded_gt)
    
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        if np.sum(pred[s:e]) > 0:
            pred[s:e] = 1

    return pred

def refine_threshold_channel_wise(scores, gt, init_threshold, window=0.15):
    if window == 0 or window == 0.0:
        best_thresh = init_threshold
    else:
        def neg_f1(thresh):
            pred = (scores > thresh).astype(int)
            pred = adjust_anomaly_channel_wise(pred, gt)
            return -f1_score(gt.ravel(), pred.ravel())

        search_bounds = (init_threshold, init_threshold + window)
        result = minimize_scalar(
            neg_f1,
            bounds=search_bounds,
            method='bounded'
        )
        best_thresh = result.x
    
    preds = (scores > best_thresh).astype(int)
    preds = adjust_anomaly_channel_wise(preds, gt)
    
    flat_gt = gt.ravel()
    flat_preds = preds.ravel()
    
    best_f1 = f1_score(flat_gt, flat_preds)
    best_precision = precision_score(flat_gt, flat_preds, zero_division=0)
    best_recall = recall_score(flat_gt, flat_preds, zero_division=0)
    accuracy = accuracy_score(flat_gt, flat_preds)
    
    tn, fp, fn, tp = confusion_matrix(flat_gt, flat_preds).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ap = average_precision_score(flat_gt, scores.ravel())

    return best_thresh, accuracy, best_precision, best_recall, best_f1, fpr, ap
   
def refine_threshold_with_metrics(scores, gt, init_threshold, window=0.002):
    def neg_f1(thresh):
        pred = scores > thresh
        adjust_anomaly(pred, gt)
        return -f1_score(gt, pred)
    if window == 0:
        best_thresh = init_threshold
        preds = scores > init_threshold
        adjust_anomaly(preds, gt)
        accuracy = accuracy_score(gt, preds)
        best_f1 = f1_score(gt, preds)
        best_precision = precision_score(gt, preds)
        best_recall = recall_score(gt, preds)
        tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
        ap = average_precision_score(gt, preds)
        fpr = fp / (fp + tn)
    else:
        result = minimize_scalar(
            neg_f1,
            bounds=(init_threshold, init_threshold+window),
            method='bounded'
        )
        best_thresh = result.x
        preds = scores > best_thresh
        adjust_anomaly(preds, gt)
        best_f1 = f1_score(gt, preds)
        best_precision = precision_score(gt, preds)
        best_recall = recall_score(gt, preds)
        accuracy = accuracy_score(gt, preds)
        tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
        fpr = fp / (fp + tn)
        ap = average_precision_score(gt, preds)

    return best_thresh,accuracy, best_precision, best_recall, best_f1, fpr, ap