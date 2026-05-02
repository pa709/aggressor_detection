

import argparse
import os
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Colours  (BGR for OpenCV)
# ──────────────────────────────────────────────────────────────────────────────

COLOR_AGGRESSOR     = (57,  41, 237)   # #ED2939
COLOR_NON_AGGRESSOR = (0, 255, 255)    # #FFFF00
FONT                = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE          = 0.55
THICKNESS           = 2

# ──────────────────────────────────────────────────────────────────────────────
# Skip overrides
# ──────────────────────────────────────────────────────────────────────────────

SKIP_OVERRIDES = {
    "F_95_0_0_0_0"                     : 180,
    "F_39_1_0_0_0"                     :  60,
    "F_127_1_2_0_0_0_53_to_1_03_clean" :   0,
    "F_135_1_2_0_0_0_00_to_0_05_clean" : -10,
}

# ──────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────────────────────────────────────

LR           = 1e-3
WEIGHT_DECAY = 1e-3
MAX_EPOCHS   = 500
PATIENCE     = 50
DROPOUT_CONV = 0.40
DROPOUT_HEAD = 0.40
SEED         = 42

# ──────────────────────────────────────────────────────────────────────────────
# TCN model — numpy, input channels = 15  (pairwise features)
# ──────────────────────────────────────────────────────────────────────────────

def conv1d_forward(x, W, b, dilation, padding):
    B, Cin, T = x.shape
    Cout, _, K = W.shape
    x_pad = np.pad(x, ((0,0),(0,0),(padding, padding)), mode="constant")
    T_out = T
    out   = np.zeros((B, Cout, T_out), dtype=np.float32)
    for k in range(K):
        offset = k * dilation
        out += np.einsum("bct,oc->bot", x_pad[:, :, offset: offset + T_out], W[:, :, k])
    out += b[np.newaxis, :, np.newaxis]
    return out, x_pad


def conv1d_backward(grad_out, x_pad, W, dilation, padding):
    B, Cout, T_out = grad_out.shape
    _, _, K = W.shape
    dx_pad = np.zeros_like(x_pad)
    dW     = np.zeros_like(W)
    db     = grad_out.sum(axis=(0, 2))
    for k in range(K):
        offset = k * dilation
        sl = slice(offset, offset + T_out)
        dW[:, :, k]   += np.einsum("bot,bct->oc", grad_out, x_pad[:, :, sl])
        dx_pad[:, :, sl] += np.einsum("bot,oc->bct", grad_out, W[:, :, k])
    dx = dx_pad[:, :, padding: padding + T_out] if padding > 0 else dx_pad
    return dx, dW, db


def relu_forward(x):       return np.maximum(0.0, x), x
def relu_backward(g, pre): return g * (pre > 0).astype(np.float32)

def dropout_forward(x, rate, training):
    if not training or rate == 0.0: return x, None
    mask = (np.random.rand(*x.shape) > rate).astype(np.float32) / (1.0 - rate)
    return x * mask, mask

def dropout_backward(g, mask): return g * mask if mask is not None else g

def gap_forward(x):     return x.mean(axis=2)
def gap_backward(g, T): return np.repeat(g[:, :, np.newaxis], T, axis=2) / T

def linear_forward(x, W, b): return x @ W.T + b

def linear_backward(grad, x, W):
    return grad @ W, grad.T @ x, grad.sum(axis=0)

def bce_with_logits_loss(logits, targets):
    return (np.log1p(np.exp(-np.abs(logits)))
            + np.maximum(logits, 0) - targets * logits).mean()

def bce_backward(logits, targets):
    return (1.0 / (1.0 + np.exp(-logits)) - targets) / len(logits)


def init_params(in_channels=15):
    """He initialisation. in_channels=15 for pairwise features."""
    def he(shape):
        fan_in = shape[1] * shape[2] if len(shape) == 3 else shape[1]
        return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)
    return {
        "W1": he((16, in_channels, 3)), "b1": np.zeros(16, dtype=np.float32),
        "W2": he((32, 16, 3)),          "b2": np.zeros(32, dtype=np.float32),
        "W3": he((32, 32, 3)),          "b3": np.zeros(32, dtype=np.float32),
        "Wh": he((1, 32, 1)).reshape(1, 32), "bh": np.zeros(1, dtype=np.float32),
    }


def forward(X, params, dr_conv, dr_head, training):
    cache = {}

    h, xp = conv1d_forward(X,  params["W1"], params["b1"], dilation=1, padding=1)
    cache["xp1"] = xp
    h, pre = relu_forward(h);  cache["pre1"] = pre
    h, m   = dropout_forward(h, dr_conv, training); cache["m1"] = m

    h, xp = conv1d_forward(h,  params["W2"], params["b2"], dilation=2, padding=2)
    cache["xp2"] = xp
    h, pre = relu_forward(h);  cache["pre2"] = pre
    h, m   = dropout_forward(h, dr_conv, training); cache["m2"] = m

    h, xp = conv1d_forward(h,  params["W3"], params["b3"], dilation=4, padding=4)
    cache["xp3"] = xp
    h, pre = relu_forward(h);  cache["pre3"] = pre
    h, m   = dropout_forward(h, dr_conv, training); cache["m3"] = m

    T = h.shape[2]; cache["T"] = T
    pool = gap_forward(h)
    pool, m = dropout_forward(pool, dr_head, training); cache["mh"] = m
    cache["pool"] = pool
    logits = linear_forward(pool, params["Wh"], params["bh"]).squeeze(-1)
    return logits, cache


def backward(logits, targets, params, cache):
    grads = {}
    g = bce_backward(logits, targets)[:, np.newaxis]
    g, grads["Wh"], grads["bh"] = linear_backward(g, cache["pool"], params["Wh"])
    g = dropout_backward(g, cache["mh"])
    g = gap_backward(g, cache["T"])
    g = dropout_backward(g, cache["m3"])
    g = relu_backward(g, cache["pre3"])
    g, grads["W3"], grads["b3"] = conv1d_backward(g, cache["xp3"], params["W3"], 4, 4)
    g = dropout_backward(g, cache["m2"])
    g = relu_backward(g, cache["pre2"])
    g, grads["W2"], grads["b2"] = conv1d_backward(g, cache["xp2"], params["W2"], 2, 2)
    g = dropout_backward(g, cache["m1"])
    g = relu_backward(g, cache["pre1"])
    _, grads["W1"], grads["b1"] = conv1d_backward(g, cache["xp1"], params["W1"], 1, 1)
    return grads


def init_adam(params):
    return ({k: np.zeros_like(v) for k, v in params.items()},
            {k: np.zeros_like(v) for k, v in params.items()})


def adam_step(params, grads, m, v, t, lr, wd,
              beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1
    for k in params:
        g = grads[k]
        m[k] = beta1 * m[k] + (1 - beta1) * g
        v[k] = beta2 * v[k] + (1 - beta2) * g ** 2
        mh = m[k] / (1 - beta1 ** t)
        vh = v[k] / (1 - beta2 ** t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps) + lr * wd * params[k]
    return t


def fit_normaliser(X):
    mu  = X.mean(axis=(0, 2), keepdims=True)
    std = X.std(axis=(0, 2),  keepdims=True) + 1e-8
    return mu, std

def normalise(X, mu, std): return (X - mu) / std


def train_model(X_all, y_all):
    """Train pairwise TCN on full dataset. Returns (params, mu, std)."""
    np.random.seed(SEED)
    mu, std = fit_normaliser(X_all)
    X_n     = normalise(X_all, mu, std)
    y       = y_all.astype(np.float32)

    params         = init_params(in_channels=15)
    m_adam, v_adam = init_adam(params)
    t              = 0
    best_loss      = np.inf
    best_params    = {k: v.copy() for k, v in params.items()}
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        logits, cache = forward(X_n, params, DROPOUT_CONV, DROPOUT_HEAD, training=True)
        loss          = bce_with_logits_loss(logits, y)
        grads         = backward(logits, y, params, cache)
        t             = adam_step(params, grads, m_adam, v_adam, t, LR, WEIGHT_DECAY)

        if loss < best_loss - 1e-5:
            best_loss      = loss
            best_params    = {k: v.copy() for k, v in params.items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stop at epoch {epoch}  (best loss {best_loss:.4f})")
                break

    return best_params, mu, std


def run_inference(X, params, mu, std):
    """X: (N, 15, 30) → int array 0/1."""
    X_n = normalise(X, mu, std)
    logits, _ = forward(X_n, params, 0.0, 0.0, training=False)
    return (logits >= 0.0).astype(int)


# ──────────────────────────────────────────────────────────────────────────────
# XML parsing — Format A and B  (identical to non-pairwise script)
# ──────────────────────────────────────────────────────────────────────────────

def detect_format(xml_path):
    root = ET.parse(xml_path).getroot()
    for tr in root.findall("track"):
        if tr.get("label", "").strip().lower() != "person":
            return "A"
    return "B"


def normalise_role(raw):
    raw = raw.strip().lower()
    if raw in ("aggressor", "agressor"):          return "aggressor"
    if raw in ("non-aggressor", "non_aggressor"): return "non-aggressor"
    return raw


def parse_xml(xml_path):
    """
    Returns:
        tracks : {track_id_str: {frame_num: (xtl, ytl, xbr, ybr, outside)}}
        fmt    : 'A' or 'B'
        fa     : first aggressor frame (int)
    """
    fmt  = detect_format(xml_path)
    root = ET.parse(xml_path).getroot()
    tracks = {}

    if fmt == "A":
        for tr in root.findall("track"):
            tid   = str(tr.get("id"))
            label = normalise_role(tr.get("label", ""))
            frames = {}
            for b in tr.findall("box"):
                fn = int(b.get("frame"))
                frames[fn] = (
                    float(b.get("xtl")), float(b.get("ytl")),
                    float(b.get("xbr")), float(b.get("ybr")),
                    int(b.get("outside", 0))
                )
            tracks[tid] = {"label": label, "frames": frames}

        fa = min(
            fn for t in tracks.values() if t["label"] == "aggressor"
            for fn, box in t["frames"].items() if box[4] == 0
        )

    else:  # Format B
        for tr in root.findall("track"):
            tid    = str(tr.get("id"))
            frames = {}
            for b in tr.findall("box"):
                fn      = int(b.get("frame"))
                outside = int(b.get("outside", 0))
                role    = next(
                    (normalise_role(a.text or "")
                     for a in b.findall("attribute")
                     if a.get("name") == "role"),
                    None
                )
                frames[fn] = (
                    float(b.get("xtl")), float(b.get("ytl")),
                    float(b.get("xbr")), float(b.get("ybr")),
                    outside, role
                )
            tracks[tid] = {"frames": frames}

        fa = min(
            fn for t in tracks.values()
            for fn, box in t["frames"].items()
            if box[4] == 0 and box[5] == "aggressor"
        )

    return tracks, fmt, fa


def get_box(tracks, track_id_str, frame_num):
    """Returns (xtl, ytl, xbr, ybr) or None. Handles merged IDs like '1+2'."""
    for tid in [t.strip() for t in track_id_str.split("+")]:
        if tid not in tracks:
            continue
        if frame_num in tracks[tid]["frames"]:
            box = tracks[tid]["frames"][frame_num]
            if box[4] == 0:
                return box[0], box[1], box[2], box[3]
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Video rendering
# ──────────────────────────────────────────────────────────────────────────────

def draw_box(frame, xtl, ytl, xbr, ybr, color, label_text):
    x1, y1, x2, y2 = int(xtl), int(ytl), int(xbr), int(ybr)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS)
    (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, THICKNESS)
    label_y = max(y1 - 4, th + 4)
    cv2.rectangle(frame,
                  (x1, label_y - th - baseline - 2),
                  (x1 + tw + 4, label_y + 2),
                  color, -1)
    cv2.putText(frame, label_text,
                (x1 + 2, label_y - baseline),
                FONT, FONT_SCALE, (0, 0, 0), THICKNESS - 1, cv2.LINE_AA)


def render_pair_video(video_path, tracks, agg_tid, non_tid,
                      pred_agg, pred_non, ws, we, out_path):
    """
    pred_agg : model prediction for the aggressor track's perspective (0 or 1)
    pred_non : model prediction for the non-aggressor track's perspective (0 or 1)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    track_draw = [
        (agg_tid, pred_agg),
        (non_tid, pred_non),
    ]

    for fn in range(ws, we + 1):
        if fn >= total:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            break

        for tid_str, pred in track_draw:
            box = get_box(tracks, tid_str, fn)
            if box is None:
                continue
            xtl, ytl, xbr, ybr = box
            if pred == 1:
                color = COLOR_AGGRESSOR
                label = "AGGRESSOR"
            else:
                color = COLOR_NON_AGGRESSOR
                label = "non-aggressor"
            draw_box(frame, xtl, ytl, xbr, ybr, color, label)

        writer.write(frame)

    cap.release()
    writer.release()


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_npz_dict(npz_args):
    result = {}
    for entry in npz_args:
        if ":" not in entry:
            raise ValueError(f"Expected ANNOTATOR:PATH, got: '{entry}'")
        annotator, path = entry.split(":", 1)
        result[annotator.strip()] = path.strip()
    return result


def load_all_npz(npz_files):
    """
    Returns:
        X_all   : (N, 15, 30)
        y_all   : (N,) float  1=aggressor perspective, 0=non-aggressor perspective
        records : list of dicts — one per sample (two per pair)
    """
    records = []
    for ann, path in npz_files.items():
        d    = np.load(path, allow_pickle=True)
        feat = d["features"].astype(np.float32)   # (N, 30, 15)
        feat = feat.transpose(0, 2, 1)             # → (N, 15, 30)
        lbls    = d["labels"].astype(int)
        vids    = d["video_names"]
        agg_ids = d["agg_track_ids"]
        non_ids = d["non_agg_track_ids"]
        pairs   = d["pair_ids"]

        for i in range(len(lbls)):
            records.append({
                "annotator"  : ann,
                "video_name" : str(vids[i]),
                "agg_track"  : str(agg_ids[i]),
                "non_track"  : str(non_ids[i]),
                "pair_id"    : int(pairs[i]),
                "label"      : int(lbls[i]),   # 1 = this sample is agg perspective
                "features"   : feat[i],
            })

    X_all = np.stack([r["features"] for r in records])
    y_all = np.array([r["label"]    for r in records], dtype=np.float32)
    return X_all, y_all, records


# ──────────────────────────────────────────────────────────────────────────────
# File finders
# ──────────────────────────────────────────────────────────────────────────────

def find_xml(xml_dir, video_name):
    """Search xml_dir and all subfolders recursively for the annotation XML.
    Tries both the full video_name and a stem with _clean stripped."""
    stems = [video_name]
    if "_clean" in video_name:
        stems.append(video_name.replace("_clean", ""))

    for dirpath, _, filenames in os.walk(xml_dir):
        for stem in stems:
            candidate = stem + "_annotations.xml"
            if candidate in filenames:
                return os.path.join(dirpath, candidate)
    return None


def find_video(videos_dirs, video_name):
    """Search each directory in videos_dirs for video_name.mp4 (or .avi).
    Handles clean filenames by stripping _clean and timestamp suffixes."""
    stem = video_name
    for suffix in ["_clean", "_annotations"]:
        if suffix in stem:
            stem = stem[:stem.index(suffix)]
    stem = re.sub(r"_\d+_\d+_to_\d+_\d+$", "", stem)
    stem = re.sub(r"_\d+_to_\d+_\d+$",     "", stem)

    for d in videos_dirs:
        for name in (video_name, stem):
            for ext in (".mp4", ".avi", ".mov", ".MP4"):
                p = os.path.join(d, name + ext)
                if os.path.exists(p):
                    return p
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualize pairwise TCN aggressor predictions on video."
    )
    parser.add_argument("--npz", nargs="+", required=True,
                        metavar="ANNOTATOR:PATH",
                        help="Pairwise NPZ files: --npz annotator1:/path/annotator1.npz ...")
    parser.add_argument("--xml_dir",    required=True,
                        help="Directory containing *_annotations.xml files.")
    parser.add_argument("--videos_dir", nargs="+", required=True,
                        metavar="DIR",
                        help="One or more directories to search for .mp4 video files.")
    parser.add_argument("--out_dir",    default="./viz_pairwise_output",
                        help="Directory to write annotated videos.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────────────
    print("Loading pairwise NPZ files ...")
    npz_files = load_npz_dict(args.npz)
    X_all, y_all, records = load_all_npz(npz_files)
    n_pairs = len(records) // 2
    print(f"  {len(records)} samples  ({n_pairs} pairs)")

    # ── Train model ──────────────────────────────────────────────────────────
    print("\nTraining pairwise TCN on full dataset ...")
    params, mu, std = train_model(X_all, y_all)

    # ── Inference ────────────────────────────────────────────────────────────
    print("\nRunning inference ...")
    preds = run_inference(X_all, params, mu, std)
    for r, pred in zip(records, preds):
        r["pred"] = int(pred)

    acc = float((preds == y_all.astype(int)).mean())
    print(f"  Training accuracy: {acc:.3f}")

    # ── Group into pairs: (video_name, pair_id) → {agg_sample, non_sample} ─
    pair_groups = defaultdict(dict)
    for r in records:
        key = (r["video_name"], r["pair_id"])
        if r["label"] == 1:
            pair_groups[key]["agg"] = r
        else:
            pair_groups[key]["non"] = r

    # ── Render one video per pair ────────────────────────────────────────────
    print(f"\nRendering {len(pair_groups)} pair videos ...")
    rendered = 0
    skipped  = 0

    for (video_name, pair_id), pair in sorted(pair_groups.items()):
        if "agg" not in pair or "non" not in pair:
            print(f"  [SKIP] {video_name} pair={pair_id}: incomplete pair in NPZ")
            skipped += 1
            continue

        agg_r = pair["agg"]
        non_r = pair["non"]
        agg_tid = agg_r["agg_track"]
        non_tid = agg_r["non_track"]

        print(f"\n  {video_name}  pair={pair_id}  "
              f"agg_track={agg_tid}  non_track={non_tid}")

        # Find XML
        xml_path = find_xml(args.xml_dir, video_name)
        if xml_path is None:
            print(f"    [SKIP] XML not found in {args.xml_dir}")
            skipped += 1
            continue

        # Find video
        video_path = find_video(args.videos_dir, video_name)
        if video_path is None:
            print(f"    [SKIP] Video not found in any of: {args.videos_dir}")
            skipped += 1
            continue

        # Parse XML
        try:
            tracks, fmt, fa = parse_xml(xml_path)
        except Exception as e:
            print(f"    [SKIP] XML parse error: {e}")
            skipped += 1
            continue

        skip = SKIP_OVERRIDES.get(video_name, 10)
        ws   = fa + skip
        we   = ws + 29
        print(f"    Format={fmt}  fa={fa}  skip={skip}  window=[{ws}..{we}]")

        # Model predictions for each perspective
        pred_agg = agg_r["pred"]   # prediction on agg track's perspective features
        pred_non = non_r["pred"]   # prediction on non-agg track's perspective features

        def label_str(p): return "AGGRESSOR" if p == 1 else "non-aggressor"
        def gt_str(label): return "aggressor" if label == 1 else "non-aggressor"
        def match(p, l): return "✓" if p == l else "✗"

        print(f"    agg_track  {agg_tid:>6s}  GT=aggressor      "
              f"Pred={label_str(pred_agg):<14s}  {match(pred_agg, 1)}")
        print(f"    non_track  {non_tid:>6s}  GT=non-aggressor  "
              f"Pred={label_str(pred_non):<14s}  {match(pred_non, 0)}")

        # Render
        safe_video = re.sub(r"[^\w\-]", "_", video_name)
        out_name   = f"{safe_video}_pair{pair_id}_predicted.mp4"
        out_path   = os.path.join(args.out_dir, out_name)

        render_pair_video(video_path, tracks, agg_tid, non_tid,
                          pred_agg, pred_non, ws, we, out_path)
        print(f"    Saved → {out_path}")
        rendered += 1

    print(f"\n{'─'*55}")
    print(f"Rendered : {rendered} pair videos")
    print(f"Skipped  : {skipped}")
    print(f"Output   : {args.out_dir}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
