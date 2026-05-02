

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, confusion_matrix)

# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="TCN pairwise aggressor classifier — LOAO cross-validation."
    )
    parser.add_argument(
        "--npz", nargs="+", required=True,
        metavar="ANNOTATOR:PATH",
        help=(
            "One or more annotator NPZ files in ANNOTATOR:PATH format. "
            "Example: --npz annotator1:/data/annotator1.npz annotator2:/data/annotator2.npz"
        ),
    )
    parser.add_argument(
        "--out_dir", default="./tcn_results",
        help="Directory for output plots (default: ./tcn_results).",
    )
    return parser.parse_args()


def build_npz_dict(npz_args):
    result = {}
    for entry in npz_args:
        if ":" not in entry:
            raise ValueError(f"Expected ANNOTATOR:PATH, got: '{entry}'")
        annotator, path = entry.split(":", 1)
        result[annotator.strip()] = path.strip()
    return result

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

LR           = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS   = 500
PATIENCE     = 50        # early stopping on val loss
DROPOUT_CONV = 0.40
DROPOUT_HEAD = 0.40
SEED         = 42

np.random.seed(SEED)

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data(npz_files):
    data = {}
    for name, path in npz_files.items():
        d = np.load(path, allow_pickle=True)
        # features: (N, 30, 15) → transpose to (N, 15, 30) for Conv1d
        X = d["features"].astype(np.float32).transpose(0, 2, 1)
        y = d["labels"].astype(np.float32)
        data[name] = (X, y)
        print(f"  {name:8s}: {X.shape}  "
              f"pos={int(y.sum())} neg={int((y==0).sum())}")
    return data

# ──────────────────────────────────────────────────────────────────────────────
# Feature normalisation (z-score per channel, fit on train)
# ──────────────────────────────────────────────────────────────────────────────

def fit_normaliser(X_train):
    # X: (N, 15, 30) → compute mean/std over N and T
    mu  = X_train.mean(axis=(0, 2), keepdims=True)   # (1, 15, 1)
    std = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
    return mu, std

def apply_normaliser(X, mu, std):
    return (X - mu) / std

# ──────────────────────────────────────────────────────────────────────────────
# Dilated Conv1d — forward and backward
# ──────────────────────────────────────────────────────────────────────────────

def conv1d_forward(x, W, b, dilation, padding):
    """
    x : (B, Cin, T)
    W : (Cout, Cin, K)
    b : (Cout,)
    returns out (B, Cout, T), x_pad (B, Cin, T+2*padding)
    """
    B, Cin, T = x.shape
    Cout, _, K = W.shape

    x_pad = np.pad(x, ((0,0),(0,0),(padding, padding)), mode="constant")
    T_out = T   # "same" padding guarantees this

    out = np.zeros((B, Cout, T_out), dtype=np.float32)
    for k_idx in range(K):
        offset = k_idx * dilation
        # x_pad slice: (B, Cin, T_out)
        out += np.einsum("bct,oc->bot",
                         x_pad[:, :, offset: offset + T_out],
                         W[:, :, k_idx])
    out += b[np.newaxis, :, np.newaxis]
    return out, x_pad


def conv1d_backward(grad_out, x_pad, W, dilation, padding):
    """
    grad_out : (B, Cout, T_out)
    x_pad    : (B, Cin, T_pad)
    W        : (Cout, Cin, K)
    returns dx (B, Cin, T), dW (Cout, Cin, K), db (Cout,)
    """
    B, Cout, T_out = grad_out.shape
    _, _, K = W.shape

    dx_pad = np.zeros_like(x_pad)
    dW     = np.zeros_like(W)
    db     = grad_out.sum(axis=(0, 2))

    for k_idx in range(K):
        offset = k_idx * dilation
        sl = slice(offset, offset + T_out)

        # dW[:, :, k_idx] = einsum('bot,bct->oc', grad_out, x_pad[:,:,sl])
        dW[:, :, k_idx] = np.einsum("bot,bct->oc",
                                     grad_out,
                                     x_pad[:, :, sl])
        # dx_pad[:,:,sl] += einsum('bot,oc->bct', grad_out, W[:,:,k_idx])
        dx_pad[:, :, sl] += np.einsum("bot,oc->bct",
                                       grad_out,
                                       W[:, :, k_idx])

    dx = dx_pad[:, :, padding: padding + T_out] if padding > 0 else dx_pad
    return dx, dW, db


# ──────────────────────────────────────────────────────────────────────────────
# Activation / pooling helpers
# ──────────────────────────────────────────────────────────────────────────────

def relu_forward(x):
    return np.maximum(0.0, x), x

def relu_backward(grad, x_pre):
    return grad * (x_pre > 0).astype(np.float32)

def dropout_forward(x, rate, training):
    if not training or rate == 0.0:
        return x, None
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32) / (1.0 - rate)
    return x * mask, mask

def dropout_backward(grad, mask, rate):
    if mask is None:
        return grad
    return grad * mask

def gap_forward(x):
    """Global average pool: (B, C, T) → (B, C)"""
    return x.mean(axis=2)

def gap_backward(grad, T):
    """grad: (B, C) → (B, C, T)"""
    return np.repeat(grad[:, :, np.newaxis], T, axis=2) / T

def linear_forward(x, W, b):
    """x: (B, Cin) → (B, Cout)"""
    return x @ W.T + b

def linear_backward(grad, x, W):
    """
    grad : (B, Cout)
    returns dx (B, Cin), dW (Cout, Cin), db (Cout,)
    """
    dx = grad @ W
    dW = grad.T @ x
    db = grad.sum(axis=0)
    return dx, dW, db

def bce_with_logits_loss(logits, targets):
    """logits: (B,)  targets: (B,)"""
    # numerically stable: log(1+exp(z)) - y*z
    loss = np.log1p(np.exp(-np.abs(logits))) + np.maximum(logits, 0) - targets * logits
    return loss.mean()

def bce_backward(logits, targets):
    """d_loss/d_logits = sigmoid(logits) - targets"""
    sig = 1.0 / (1.0 + np.exp(-logits))
    return (sig - targets) / len(logits)

# ──────────────────────────────────────────────────────────────────────────────
# Model: parameter initialisation
# ──────────────────────────────────────────────────────────────────────────────

def init_params():
    """
    Returns dict of all trainable parameters.
    Conv layers use He initialisation.
    """
    def he(shape):
        fan_in = shape[1] * shape[2] if len(shape) == 3 else shape[1]
        return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

    params = {
        # Block 1: Conv1d(15→16, k=3, d=1)
        "W1": he((16, 15, 3)),  "b1": np.zeros(16, dtype=np.float32),
        # Block 2: Conv1d(16→32, k=3, d=2)
        "W2": he((32, 16, 3)),  "b2": np.zeros(32, dtype=np.float32),
        # Block 3: Conv1d(32→32, k=3, d=4)
        "W3": he((32, 32, 3)),  "b3": np.zeros(32, dtype=np.float32),
        # Head: Linear(32→1)
        "Wh": he((1,  32, 1)).reshape(1, 32),  "bh": np.zeros(1, dtype=np.float32),
    }
    return params


# ──────────────────────────────────────────────────────────────────────────────
# Full forward pass — returns logits and cache for backward
# ──────────────────────────────────────────────────────────────────────────────

def forward(X, params, dropout_rate_conv, dropout_rate_head, training):
    """
    X: (B, 15, 30)
    Returns: logits (B,), cache dict
    """
    cache = {}

    # Block 1
    h, xp = conv1d_forward(X,  params["W1"], params["b1"], dilation=1, padding=1)
    cache["xp1"] = xp
    h, h_pre = relu_forward(h);            cache["pre1"] = h_pre
    h, mask   = dropout_forward(h, dropout_rate_conv, training); cache["m1"] = mask

    # Block 2
    h, xp = conv1d_forward(h,  params["W2"], params["b2"], dilation=2, padding=2)
    cache["xp2"] = xp;  cache["in2"] = h  # save pre-relu input for backward
    h, h_pre = relu_forward(h);            cache["pre2"] = h_pre
    h, mask   = dropout_forward(h, dropout_rate_conv, training); cache["m2"] = mask

    # Block 3
    h, xp = conv1d_forward(h,  params["W3"], params["b3"], dilation=4, padding=4)
    cache["xp3"] = xp
    h, h_pre = relu_forward(h);            cache["pre3"] = h_pre
    h, mask   = dropout_forward(h, dropout_rate_conv, training); cache["m3"] = mask

    # Global average pool
    T = h.shape[2]
    pool = gap_forward(h);                  cache["T"] = T;  cache["pre_pool"] = h

    # Dropout before head
    pool, mask = dropout_forward(pool, dropout_rate_head, training); cache["mh"] = mask

    # Linear head
    cache["pool"] = pool
    logits = linear_forward(pool, params["Wh"], params["bh"]).squeeze(-1)   # (B,)

    return logits, cache


# ──────────────────────────────────────────────────────────────────────────────
# Full backward pass
# ──────────────────────────────────────────────────────────────────────────────

def backward(logits, targets, params, cache):
    """Returns dict of gradients with same keys as params."""
    grads = {}

    # Head backward
    g = bce_backward(logits, targets)[:, np.newaxis]        # (B, 1)
    g, grads["Wh"], grads["bh"] = linear_backward(g, cache["pool"], params["Wh"])

    # Dropout head backward
    g = dropout_backward(g, cache["mh"], rate=0)            # mask already baked in

    # GAP backward
    g = gap_backward(g, cache["T"])                         # (B, 32, 30)

    # Block 3 backward
    g = dropout_backward(g, cache["m3"], rate=0)
    g = relu_backward(g, cache["pre3"])
    g, grads["W3"], grads["b3"] = conv1d_backward(g, cache["xp3"],
                                                   params["W3"],
                                                   dilation=4, padding=4)

    # Block 2 backward
    g = dropout_backward(g, cache["m2"], rate=0)
    g = relu_backward(g, cache["pre2"])
    g, grads["W2"], grads["b2"] = conv1d_backward(g, cache["xp2"],
                                                   params["W2"],
                                                   dilation=2, padding=2)

    # Block 1 backward
    g = dropout_backward(g, cache["m1"], rate=0)
    g = relu_backward(g, cache["pre1"])
    _, grads["W1"], grads["b1"] = conv1d_backward(g, cache["xp1"],
                                                   params["W1"],
                                                   dilation=1, padding=1)
    return grads


# ──────────────────────────────────────────────────────────────────────────────
# AdamW optimiser state
# ──────────────────────────────────────────────────────────────────────────────

def init_adam(params):
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    return m, v


def adam_step(params, grads, m, v, t, lr, weight_decay,
              beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
        m_hat = m[k] / (1 - beta1 ** t)
        v_hat = v[k] / (1 - beta2 ** t)
        # AdamW: apply weight decay directly to params
        params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps) + lr * weight_decay * params[k]
    return t


# ──────────────────────────────────────────────────────────────────────────────
# Predict (no dropout)
# ──────────────────────────────────────────────────────────────────────────────

def predict(X, params):
    logits, _ = forward(X, params, dropout_rate_conv=0.0,
                        dropout_rate_head=0.0, training=False)
    return (logits >= 0.0).astype(int)

def predict_proba(X, params):
    logits, _ = forward(X, params, dropout_rate_conv=0.0,
                        dropout_rate_head=0.0, training=False)
    return 1.0 / (1.0 + np.exp(-logits))


# ──────────────────────────────────────────────────────────────────────────────
# Train one fold
# ──────────────────────────────────────────────────────────────────────────────

def train_fold(X_tr, y_tr, X_val, y_val):
    np.random.seed(SEED)
    params    = init_params()
    m_adam, v_adam = init_adam(params)
    t = 0

    best_val_loss  = np.inf
    best_params    = {k: v.copy() for k, v in params.items()}
    patience_count = 0

    train_losses, val_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # Forward (training mode)
        logits, cache = forward(X_tr, params,
                                dropout_rate_conv=DROPOUT_CONV,
                                dropout_rate_head=DROPOUT_HEAD,
                                training=True)
        tr_loss = bce_with_logits_loss(logits, y_tr)

        # Backward
        grads = backward(logits, y_tr, params, cache)

        # AdamW update
        t = adam_step(params, grads, m_adam, v_adam, t,
                      lr=LR, weight_decay=WEIGHT_DECAY)

        # Validation loss (no dropout)
        val_logits, _ = forward(X_val, params,
                                dropout_rate_conv=0.0,
                                dropout_rate_head=0.0,
                                training=False)
        vl = bce_with_logits_loss(val_logits, y_val)

        train_losses.append(float(tr_loss))
        val_losses.append(float(vl))

        # Early stopping
        if vl < best_val_loss - 1e-5:
            best_val_loss  = vl
            best_params    = {k: v.copy() for k, v in params.items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    Early stop at epoch {epoch}  "
                      f"(best val loss {best_val_loss:.4f})")
                break

    return best_params, train_losses, val_losses


# ──────────────────────────────────────────────────────────────────────────────
# LOAO cross-validation
# ──────────────────────────────────────────────────────────────────────────────

def run_loao(data):
    annotators = list(data.keys())
    results    = {}
    all_curves = {}

    for test_ann in annotators:
        print(f"\n{'─'*50}")
        print(f"Fold: test = {test_ann}")

        # Build train / val splits
        X_tr_list, y_tr_list = [], []
        for ann, (X, y) in data.items():
            if ann != test_ann:
                X_tr_list.append(X)
                y_tr_list.append(y)

        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)
        X_val, y_val = data[test_ann]

        print(f"  Train: {X_tr.shape[0]} samples  |  Val: {X_val.shape[0]} samples")

        # Normalise
        mu, std = fit_normaliser(X_tr)
        X_tr_n  = apply_normaliser(X_tr,  mu, std)
        X_val_n = apply_normaliser(X_val, mu, std)

        # Train
        best_params, tr_losses, val_losses = train_fold(X_tr_n, y_tr, X_val_n, y_val)

        # Evaluate
        y_pred = predict(X_val_n, best_params)
        acc  = accuracy_score(y_val,  y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec  = recall_score(y_val,   y_pred, zero_division=0)
        f1   = f1_score(y_val,       y_pred, zero_division=0)
        cm   = confusion_matrix(y_val, y_pred)

        print(f"  Acc={acc:.3f}  Prec={prec:.3f}  Rec={rec:.3f}  F1={f1:.3f}")
        print(f"  Confusion matrix:\n  {cm}")

        results[test_ann] = dict(acc=acc, prec=prec, rec=rec, f1=f1, cm=cm,
                                 y_val=y_val, y_pred=y_pred)
        all_curves[test_ann] = (tr_losses, val_losses)

    return results, all_curves


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_curves(all_curves, out_dir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    annotators = list(all_curves.keys())

    for i, ann in enumerate(annotators):
        tr_losses, val_losses = all_curves[ann]
        ax = axes[i]
        ax.plot(tr_losses,  label="Train loss", color="#2196F3", lw=1.5)
        ax.plot(val_losses, label="Val loss",   color="#F44336", lw=1.5, ls="--")
        ax.set_title(f"Fold: test={ann}", fontsize=11)
        ax.set_xlabel("Epoch"); ax.set_ylabel("BCE loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[-1].axis("off")
    plt.suptitle("TCN Training Curves — LOAO Cross-Validation", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "tcn_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def plot_confusion_matrices(results, out_dir):
    annotators = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for i, ann in enumerate(annotators):
        cm = results[ann]["cm"]
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues", vmin=0)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["non-agg", "agg"]); ax.set_yticklabels(["non-agg", "agg"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Test={ann}  F1={results[ann]['f1']:.3f}", fontsize=11)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]),
                        ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    axes[-1].axis("off")
    plt.suptitle("Confusion Matrices — LOAO Cross-Validation", fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(out_dir, "tcn_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_summary_bar(results, out_dir):
    annotators = list(results.keys())
    metrics = ["acc", "prec", "rec", "f1"]
    labels  = ["Accuracy", "Precision", "Recall", "F1"]
    colors  = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

    x = np.arange(len(annotators))
    width = 0.20

    fig, ax = plt.subplots(figsize=(12, 5))
    for j, (m, lab, col) in enumerate(zip(metrics, labels, colors)):
        vals = [results[a][m] for a in annotators]
        bars = ax.bar(x + j * width, vals, width, label=lab, color=col, alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # Mean lines
    for j, (m, col) in enumerate(zip(metrics, colors)):
        mean_val = np.mean([results[a][m] for a in annotators])
        ax.axhline(mean_val, color=col, lw=1.2, ls=":", alpha=0.7)

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(annotators, fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("TCN Performance per Fold — LOAO Cross-Validation", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, "tcn_fold_performance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Summary table
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(results):
    annotators = list(results.keys())
    header = f"{'Fold':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}"
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")

    accs, precs, recs, f1s = [], [], [], []
    for ann in annotators:
        r = results[ann]
        print(f"{ann:<10} {r['acc']:>6.3f} {r['prec']:>6.3f} {r['rec']:>6.3f} {r['f1']:>6.3f}")
        accs.append(r["acc"]); precs.append(r["prec"])
        recs.append(r["rec"]); f1s.append(r["f1"])

    print(sep)
    print(f"{'mean':<10} {np.mean(accs):>6.3f} {np.mean(precs):>6.3f} "
          f"{np.mean(recs):>6.3f} {np.mean(f1s):>6.3f}")
    print(f"{'std':<10} {np.std(accs):>6.3f} {np.std(precs):>6.3f} "
          f"{np.std(recs):>6.3f} {np.std(f1s):>6.3f}")
    print(sep)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    npz_files = build_npz_dict(args.npz)
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Patch plot save paths to use out_dir
    global _OUT_DIR
    _OUT_DIR = out_dir

    print("=" * 55)
    print("TCN Pairwise Aggressor Classifier — LOAO CV")
    print("=" * 55)
    print("\nArchitecture:")
    print("  Conv1d(15→16, k=3, d=1) → ReLU → Dropout(0.4)")
    print("  Conv1d(16→32, k=3, d=2) → ReLU → Dropout(0.4)")
    print("  Conv1d(32→32, k=3, d=4) → ReLU → Dropout(0.4)")
    print("  Global Avg Pool → Dropout(0.4) → Linear(32→1)")
    print(f"\nHyperparams: lr={LR}  wd={WEIGHT_DECAY}  "
          f"dropout={DROPOUT_CONV}  patience={PATIENCE}")
    print(f"\nOutput dir: {out_dir}")

    print("\nLoading data ...")
    data = load_data(npz_files)

    print("\nRunning LOAO cross-validation ...")
    results, all_curves = run_loao(data)

    print_summary(results)

    print("\nGenerating plots ...")
    plot_training_curves(all_curves, out_dir)
    plot_confusion_matrices(results, out_dir)
    plot_summary_bar(results, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
