import argparse
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from core.config import (BATCH_SIZE, CHECKPOINTS_DIR, EARLY_STOP_PATIENCE,
                          LEARNING_RATE, MAX_EPOCHS, NUM_CLASSES)
from model.architecture import TamperNet
from model.dataset import TamperDataset
from model.evaluate import evaluate


def compute_loss(pred_mask, pred_logit, gt_mask, gt_label, pos_weight: float = 1.0):
    label_loss = nn.CrossEntropyLoss()(pred_logit, gt_label.long())

    tampered = gt_label == 2
    if tampered.any():
        mask_loss = nn.BCELoss()(pred_mask[tampered], gt_mask[tampered])
    else:
        mask_loss = torch.tensor(0.0, device=pred_logit.device)

    return label_loss + 0.5 * mask_loss


def train(on_epoch=None, resume: bool = True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Training on {device}')

    train_ds = TamperDataset(split='train')
    val_ds   = TamperDataset(split='val')
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    labels = [lbl for _, lbl in train_ds.items]
    counts = [max(labels.count(label), 1) for label in range(NUM_CLASSES)]
    print(f'Class balance: original={counts[0]} / ai-generated={counts[1]} / forged={counts[2]}')

    model = TamperNet().to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    resume_path = CHECKPOINTS_DIR / 'resume_state.pt'

    start_epoch = 1
    best_auc    = 0.0
    no_improve  = 0
    history     = []

    if resume and resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_auc    = ckpt['best_auc']
        no_improve  = ckpt.get('no_improve', 0)
        history     = ckpt.get('history', [])
        print(f'Resuming from resume_state.pt: epoch {start_epoch}, best_auc so far {best_auc:.4f}')
    else:
        print('Starting fresh training run (no resume_state.pt found, or resume=False).')

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0

        for imgs, masks, labels in train_dl:
            imgs, masks, labels = imgs.to(device), masks.to(device), labels.to(device)
            opt.zero_grad()
            pred_mask, pred_logit = model(imgs)
            loss = compute_loss(pred_mask, pred_logit, masks, labels)
            loss.backward()
            opt.step()
            train_loss += loss.item()

        metrics = evaluate(model, val_dl, device)
        sched.step(metrics['auc'])

        epoch_metrics = {'loss': round(train_loss / len(train_dl), 4), **metrics}
        print(f"Epoch {epoch:03d} | loss={epoch_metrics['loss']:.4f} "
              f"| acc={metrics.get('accuracy', float('nan')):.4f} "
              f"| prec={metrics.get('precision', float('nan')):.4f} "
              f"| rec={metrics.get('recall', float('nan')):.4f} "
              f"| auc={metrics['auc']:.4f} | iou={metrics['pixel_iou']:.4f}")

        history.append({'epoch': epoch, **epoch_metrics})
        with open(CHECKPOINTS_DIR / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        if on_epoch:
            on_epoch(epoch, epoch_metrics)

        if metrics['auc'] > best_auc:
            best_auc   = metrics['auc']
            no_improve = 0
            torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best.pt')
            print(f'  saved best model (auc={best_auc:.4f})')
        else:
            no_improve += 1

        torch.save({
            'model':      model.state_dict(),
            'optimizer':  opt.state_dict(),
            'scheduler':  sched.state_dict(),
            'epoch':      epoch,
            'best_auc':   best_auc,
            'no_improve': no_improve,
            'history':    history,
        }, resume_path)

        if no_improve >= EARLY_STOP_PATIENCE:
            print(f'Early stopping at epoch {epoch}')
            break

    torch.save(model.state_dict(), CHECKPOINTS_DIR / 'last.pt')
    if not (CHECKPOINTS_DIR / 'best.pt').exists():
        torch.save(model.state_dict(), CHECKPOINTS_DIR / 'best.pt')
        print('Saved best.pt (fallback — auc never exceeded 0.0)')

    with open(CHECKPOINTS_DIR / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f'Training done. Best AUC: {best_auc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', action='store_true',
                         help='Ignore any existing resume_state.pt and start from epoch 1.')
    args = parser.parse_args()
    train(resume=not args.fresh)
