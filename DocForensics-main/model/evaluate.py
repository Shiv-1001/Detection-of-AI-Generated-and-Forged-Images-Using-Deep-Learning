import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                              recall_score, accuracy_score, confusion_matrix)


def evaluate(model, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    all_labels, all_scores = [], []
    total_iou, n = 0.0, 0

    with torch.no_grad():
        for imgs, masks, labels in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            pred_mask, pred_logit = model(imgs)

            probs = torch.softmax(pred_logit, dim=1).cpu().numpy()

            all_scores.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())

            pred_bin = (pred_mask > 0.5).float()
            gt_bin   = (masks > 0.5).float()
            intersection = (pred_bin * gt_bin).sum()
            union        = pred_bin.sum() + gt_bin.sum() - intersection
            total_iou   += (intersection / (union + 1e-6)).item()
            n           += 1

        preds = np.argmax(all_scores, axis=1)
        auc = (roc_auc_score(all_labels, all_scores, multi_class='ovr')
            if len(set(all_labels)) == 3 else 0.0)

    # NEW: accuracy / precision / recall alongside the existing AUC / F1 / IoU,
    # so the report can cite more than a single accuracy number. Macro-averaging
    # is used (unweighted mean across the 3 classes) since the classes are not
    # guaranteed to be balanced.
    return {
        'accuracy':  round(float(accuracy_score(all_labels, preds)), 4),
        'precision': round(float(precision_score(all_labels, preds, average='macro', zero_division=0)), 4),
        'recall':    round(float(recall_score(all_labels, preds, average='macro', zero_division=0)), 4),
        'auc':       round(float(auc), 4),
        'f1':        round(float(f1_score(all_labels, preds, average='macro', zero_division=0)), 4),
        'pixel_iou': round(total_iou / max(n, 1), 4),
        'confusion_matrix': confusion_matrix(all_labels, preds, labels=[0, 1, 2]).tolist(),
    }