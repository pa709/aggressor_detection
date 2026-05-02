

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)
from sklearn.preprocessing import StandardScaler


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

# Annotator names are inferred from NPZ filenames at runtime (see load_data)

def load_data(data_dir: str):
    """Discover and load all optical_flow_features_*.npz in data_dir.
    Returns X (N,30,11), y (N,), src (N,) annotator name."""
    import glob
    pattern   = os.path.join(data_dir, 'optical_flow_features_*.npz')
    npz_files = sorted(glob.glob(pattern))
    if not npz_files:
        raise FileNotFoundError(f'No optical_flow_features_*.npz files found in: {data_dir}')
    X_all, y_all, src_all = [], [], []
    for fpath in npz_files:
        name = os.path.basename(fpath).replace('optical_flow_features_', '').replace('.npz', '')
        d    = np.load(fpath, allow_pickle=True)
        X    = d['features'].astype(np.float32)          # (N, 30, 11)
        y    = (d['labels'] == 'aggressor').astype(np.float32)
        X_all.append(X)
        y_all.append(y)
        src_all.extend([name] * len(X))
        print(f'[INFO] Loaded {len(X)} samples from {os.path.basename(fpath)}')
    return (np.concatenate(X_all),
            np.concatenate(y_all),
            np.array(src_all))


# ──────────────────────────────────────────────────────────────────────────────
# TCN architecture
# ──────────────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    """
    One TCN residual block:
      - Two dilated causal Conv1d layers with same padding
      - Residual (skip) connection with 1x1 conv if channels change
      - BatchNorm + ReLU + Dropout
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Causal padding: pad left only so no future info leaks
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size,
                                dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size,
                                dilation=dilation, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.dropout  = nn.Dropout(dropout)
        self.skip     = (nn.Conv1d(in_ch, out_ch, 1)
                         if in_ch != out_ch else nn.Identity())

    def forward(self, x):
        # x: (B, C, T)
        T   = x.size(2)
        res = self.skip(x)

        out = self.conv1(x)[..., :T]          # trim causal padding
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)[..., :T]
        out = self.bn2(out)
        out = F.relu(out + res)               # residual
        out = self.dropout(out)
        return out


class TCNClassifier(nn.Module):
    def __init__(self, in_features=11, dropout=0.3):
        super().__init__()
        self.tcn = nn.Sequential(
            TCNBlock(in_features, 32, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(32,          64, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(64,          64, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, C) -> transpose to (B, C, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.tcn(x)                       # (B, 64, T)
        x = x.mean(dim=2)                     # global average pool -> (B, 64)
        return self.head(x).squeeze(1)        # (B,) logits


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimiser.zero_grad()
        logits = model(X_b)
        loss   = criterion(logits, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()
        total_loss += loss.item() * len(y_b)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    for X_b, y_b in loader:
        logits = model(X_b.to(device))
        prob   = torch.sigmoid(logits).cpu().numpy()
        preds.extend((prob >= 0.5).astype(int).tolist())
        targets.extend(y_b.int().tolist())
    return np.array(preds), np.array(targets)


def metrics(preds, targets, label=''):
    acc  = accuracy_score(targets, preds)
    prec = precision_score(targets, preds, zero_division=0)
    rec  = recall_score(targets, preds, zero_division=0)
    f1   = f1_score(targets, preds, zero_division=0)
    cm   = confusion_matrix(targets, preds)
    tag  = f'[{label}] ' if label else ''
    print(f'  {tag}Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}')
    print(f'  Confusion matrix (rows=true, cols=pred):')
    print(f'    non-agg aggressor')
    for i, row in enumerate(cm):
        lbl = 'non-agg ' if i == 0 else 'aggress '
        print(f'    {lbl} {row}')
    return acc, prec, rec, f1


# ──────────────────────────────────────────────────────────────────────────────
# Leave-one-annotator-out cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def run_loao_cv(X, y, src, args, device):
    fold_metrics = []
    annotators   = sorted(np.unique(src).tolist())

    for val_ann in annotators:
        print(f'\n{"="*60}')
        print(f'Fold: held-out annotator = {val_ann}')
        print(f'{"="*60}')

        train_mask = src != val_ann
        val_mask   = src == val_ann

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask],   y[val_mask]

        print(f'  Train: {len(X_tr)} samples  '
              f'(agg={int(y_tr.sum())}, non={int((1-y_tr).sum())})')
        print(f'  Val  : {len(X_va)} samples  '
              f'(agg={int(y_va.sum())}, non={int((1-y_va).sum())})')

        # Normalise per-feature using training statistics
        # Shape (N, 30, 11) -> reshape to (N*30, 11) for scaler
        scaler = StandardScaler()
        shape  = X_tr.shape
        X_tr_s = scaler.fit_transform(X_tr.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
        X_va_s = scaler.transform(X_va.reshape(-1, shape[-1])).reshape(X_va.shape).astype(np.float32)

        # Class weights to handle imbalance
        pos_weight = torch.tensor(
            [(1 - y_tr).sum() / (y_tr.sum() + 1e-6)],
            dtype=torch.float32, device=device
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        # DataLoaders
        tr_ds  = TensorDataset(torch.from_numpy(X_tr_s), torch.from_numpy(y_tr))
        va_ds  = TensorDataset(torch.from_numpy(X_va_s), torch.from_numpy(y_va))
        tr_dl  = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,  drop_last=False)
        va_dl  = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)

        # Model
        model = TCNClassifier(in_features=X.shape[-1], dropout=args.dropout).to(device)
        opt   = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

        best_f1, best_state = 0.0, None
        for epoch in range(1, args.epochs + 1):
            loss = train_epoch(model, tr_dl, opt, criterion, device)
            sched.step()
            if epoch % 50 == 0 or epoch == args.epochs:
                preds, targets = evaluate(model, va_dl, device)
                f1 = f1_score(targets, preds, zero_division=0)
                print(f'  Epoch {epoch:>3d}  loss={loss:.4f}  val_f1={f1:.3f}')
                if f1 >= best_f1:
                    best_f1   = f1
                    best_state = {k: v.cpu().clone()
                                  for k, v in model.state_dict().items()}

        # Evaluate best model on val
        model.load_state_dict(best_state)
        preds, targets = evaluate(model, va_dl, device)
        print(f'\n  Best val results for fold [{val_ann}]:')
        acc, prec, rec, f1 = metrics(preds, targets)
        fold_metrics.append((val_ann, acc, prec, rec, f1))

    return fold_metrics


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',     default='.',
                        help='Directory containing the .npz files')
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--dropout',      type=float, default=0.3)
    parser.add_argument('--seed',         type=int,   default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    X, y, src = load_data(args.data_dir)
    print(f'\nLoaded {len(X)} samples  '
          f'(agg={int(y.sum())}, non-agg={int((1-y).sum())})')
    print(f'Feature shape: {X.shape}  (N x T x C)')

    fold_metrics = run_loao_cv(X, y, src, args, device)

    # Summary
    print(f'\n{"="*60}')
    print('LEAVE-ONE-ANNOTATOR-OUT SUMMARY')
    print(f'{"="*60}')
    print(f'  {"Fold":<10s}  {"Acc":>6s}  {"Prec":>6s}  {"Rec":>6s}  {"F1":>6s}')
    print(f'  {"-"*42}')
    for ann, acc, prec, rec, f1 in fold_metrics:
        print(f'  {ann:<10s}  {acc:>6.3f}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}')
    print(f'  {"-"*42}')
    avg = np.mean([m[1:] for m in fold_metrics], axis=0)
    std = np.std( [m[1:] for m in fold_metrics], axis=0)
    print(f'  {"mean":<10s}  {avg[0]:>6.3f}  {avg[1]:>6.3f}  {avg[2]:>6.3f}  {avg[3]:>6.3f}')
    print(f'  {"std":<10s}  {std[0]:>6.3f}  {std[1]:>6.3f}  {std[2]:>6.3f}  {std[3]:>6.3f}')


if __name__ == '__main__':
    main()
