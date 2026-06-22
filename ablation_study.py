"""
Ablation Study for CAFES
========================
Evaluates the contribution of three key components:
  1. Feature Channels  – raw, diff, w_mean, w_std, t_stat, z_score
  2. SEBlock            – Squeeze-and-Excitation attention
  3. Dropout            – regularisation rate

Usage
-----
  python ablation_study.py \
      --train_pos_data_folder <train_pos_dir> --train_neg_data_folder <train_neg_dir> \
      --test_pos_data_folder <test_pos_dir> --test_neg_data_folder <test_neg_dir> \
      -o <output_dir> \
      [--gpu_ids 0] [--experiments baseline_full feat_raw_only ...]

Outputs
-------
  <output_dir>/
    ablation_results.csv      – per-experiment metrics
    ablation_summary.txt      – formatted table
    ablation_<group>.pdf      – bar-chart comparisons
    <exp_name>/model.pth      – best checkpoint per experiment
"""

import os
import sys
import csv
import time
import argparse

import torch
import tqdm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from sklearn.utils import shuffle

from models.CAFES import CAFES
from dataset import Dataset, LazyTrainDataset
from preprocessor import add_features, valid_non_normalization, feature_window_size_type

# ── Constants ────────────────────────────────────────────────────────
CHANNEL_NAMES = ["raw", "diff", "w_mean", "w_std", "t_stat", "z_score"]
DEFAULT_FEATURE_WINDOW_SIZE = 3


def unique_window_sizes(window_sizes):
    if window_sizes is None:
        return [DEFAULT_FEATURE_WINDOW_SIZE]
    unique = []
    for w_len in window_sizes:
        if w_len not in unique:
            unique.append(w_len)
    return unique

# ── Channel-selection utilities ──────────────────────────────────────

def myprint(string, log):
    log.write(string + '\n')
    log.flush()
    print(string)

class ChannelSelectDataset(torch.utils.data.Dataset):
    """Wraps a base dataset and keeps only the requested channels."""
    def __init__(self, base_dataset, channel_indices):
        self.base = base_dataset
        self.ch = channel_indices
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        X, Y = self.base[idx]
        return X[self.ch], Y

def select_channels(tensor, channel_indices):
    """Select channels from (N, C, L) tensor."""
    return tensor[:, channel_indices, :]


# ── Experiment definitions ───────────────────────────────────────────

def get_experiments(window_sizes=None):
    """Return the full list of ablation configurations.

    Groups
    ------
    baseline          – full model (shared reference for every group)
    feature_channels  – progressively remove feature channels
    SEBlock           – with / without squeeze-and-excitation
    dropout           – sweep dropout rates
    """
    exps = []

    # ── BN ablation ─────────────────────────────────────────────

    exps.append(dict(
        name="rc", group="rc",
        channels=[0],
        channel_desc="raw signal",
        use_SEBlock=False, dropout=0, norm=False, use_first_bn=False,
    ))

    exps.append(dict(
        name="first_bn_disabled", group="BatchNorm",
        channels=[0, 1, 2, 3, 4, 5],
        channel_desc="All 6ch",
        use_SEBlock=True, dropout=0.2, norm=True, use_first_bn=False,
    ))

    # ── Baseline (shared) ────────────────────────────────────────────
    exps.append(dict(
        name="baseline_full",       group="baseline",
        channels=[0, 1, 2, 3, 4, 5],
        channel_desc="All 6ch",
        use_SEBlock=True, dropout=0.2, norm=True,
    ))

    # ── Feature-channel ablation ─────────────────────────────────────
    exps.append(dict(
        name="feat_no_zscore",      group="feature_channels",
        channels=[0, 1, 2, 3, 4],
        channel_desc="5ch (w/o z_score)",
        use_SEBlock=True, dropout=0.2, norm=False,
    ))
    exps.append(dict(
        name="feat_no_tstat",       group="feature_channels",
        channels=[0, 1, 2, 3, 5],
        channel_desc="5ch (w/o t_stat)",
        use_SEBlock=True, dropout=0.2, norm=True,
    ))
    exps.append(dict(
        name="feat_no_tstat_zscore", group="feature_channels",
        channels=[0, 1, 2, 3],
        channel_desc="4ch (raw+diff+w_mean+w_std)",
        use_SEBlock=True, dropout=0.2, norm=False,
    ))
    exps.append(dict(
        name="feat_raw_diff",       group="feature_channels",
        channels=[0, 1],
        channel_desc="2ch (raw+diff)",
        use_SEBlock=True, dropout=0.2, norm=False,
    ))
    exps.append(dict(
        name="feat_raw_only",       group="feature_channels",
        channels=[0],
        channel_desc="1ch (raw only)",
        use_SEBlock=True, dropout=0.2, norm=False,
    ))

    # ── SEBlock ablation ─────────────────────────────────────────────
    exps.append(dict(
        name="se_disabled",         group="SEBlock",
        channels=[0, 1, 2, 3, 4, 5],
        channel_desc="All 6ch",
        use_SEBlock=False, dropout=0.2, norm=True, use_first_bn=True,
    ))



    # ── Dropout ablation ─────────────────────────────────────────────
    for dr in [0.0, 0.1, 0.3, 0.5]:
        exps.append(dict(
            name=f"dropout_{dr}",   group="dropout",
            channels=[0, 1, 2, 3, 4, 5],
            channel_desc="All 6ch",
            use_SEBlock=True, dropout=dr, norm=True,
        ))

    for exp in exps:
        exp.setdefault("feature_window_size", DEFAULT_FEATURE_WINDOW_SIZE)

    for w_len in unique_window_sizes(window_sizes):
        if w_len == DEFAULT_FEATURE_WINDOW_SIZE:
            continue
        exps.append(dict(
            name=f"window_size_{w_len}", group="feature_window_size",
            channels=[0, 1, 2, 3, 4, 5],
            channel_desc=f"All 6ch, window={w_len}",
            use_SEBlock=True, dropout=0.2, norm=True,
            use_first_bn=True,
            feature_window_size=w_len,
        ))

    return exps


# ── Training loop ────────────────────────────────────────────────────

def train_model(model, pos_loader, neg_loader,
                pos_val_loader, neg_val_loader,
                out_dir, device,
                lr=1e-3, epochs=300, tolerance=10, log_fh=None):
    """Train *model* and return (best_val_acc, best_val_loss, time_s, early)."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda ep: 1 / (2 ** ep))

    best_acc, best_loss = 0.0, 1e7
    patience, sched_num, step = 0, 0, 0
    acc_buf, loss_buf = 0.0, 0.0
    avg_train_acc, avg_train_loss = 0.0, 0.0
    neg_iter = iter(neg_loader)
    neg_val_iter = iter(neg_val_loader)
    pos_epoch = 1

    t0 = time.time()
    while pos_epoch <= epochs:
        pbar = tqdm.tqdm(pos_loader, desc=f"epoch {pos_epoch}",
                         bar_format="{l_bar}{r_bar}")
        for pos_x, pos_y in pbar:
            step += 1
            try:
                neg_x, neg_y = next(neg_iter)
            except StopIteration:
                neg_iter = iter(neg_loader)
                neg_x, neg_y = next(neg_iter)

            x = torch.cat((pos_x, neg_x))
            y = torch.cat((pos_y, neg_y))
            x, y = shuffle(x, y)
            x, y = x.float().to(device), y.long().to(device)

            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_buf += (y == out.max(1).indices).float().mean().item()
            loss_buf += loss.item()

            if step % 50 == 0:
                avg_train_acc = round(acc_buf * 2, 2)
                avg_train_loss = round(loss_buf / 50, 4)
                acc_buf, loss_buf = 0.0, 0.0
                
                if log_fh:
                    current_lr = optimizer.param_groups[0]['lr']
                    post_fix = {
                        "pos epoch": pos_epoch,
                        "iter": step,
                        "avg_loss": avg_train_loss,
                        "avg_acc": avg_train_acc,
                        "lr": current_lr
                    }
                    print(str(post_fix), file=log_fh)
                    log_fh.flush()

            # ── Validation every 200 steps ───────────────────────────
            if step % 200 == 0:
                model.eval()
                with torch.no_grad():
                    v_acc, v_loss, vn = 0.0, 0.0, 0
                    for pv_x, pv_y in pos_val_loader:
                        vn += 1
                        try:
                            nv_x, nv_y = next(neg_val_iter)
                        except StopIteration:
                            neg_val_iter = iter(neg_val_loader)
                            nv_x, nv_y = next(neg_val_iter)
                        vx = torch.cat((pv_x, nv_x)).float().to(device)
                        vy = torch.cat((pv_y, nv_y)).long().to(device)
                        vx, vy = shuffle(vx, vy)
                        vo = model(vx)
                        v_acc += (vy == vo.max(1).indices).float().mean().item()
                        v_loss += criterion(vo, vy).item()

                    avg_v_acc = round((v_acc / vn) * 100, 2) if vn else 0
                    avg_v_loss = round(v_loss / vn, 4) if vn else 0

                    if best_loss <= avg_v_loss:
                        patience += 1
                    else:
                        patience = 0
                        best_acc, best_loss = avg_v_acc, avg_v_loss
                        torch.save(model.state_dict(),
                                   os.path.join(out_dir, "model.pth"))

                    if log_fh:
                        post_fix = {
                            "valid_loss": avg_v_loss,
                            "valid_acc": avg_v_acc,
                            "best_loss": best_loss,
                            "best_acc": best_acc,
                        }
                        print(str(post_fix), file=log_fh)
                        log_fh.flush()

                    if patience >= tolerance and sched_num < 4:
                        model.load_state_dict(
                            torch.load(os.path.join(out_dir, "model.pth")))
                        scheduler.step()
                        sched_num += 1
                        patience = 0
                    elif patience >= tolerance:
                        return best_acc, best_loss, time.time() - t0, True
                model.train()
        pos_epoch += 1

    return best_acc, best_loss, time.time() - t0, False


# ── Testing ──────────────────────────────────────────────────────────

def test_model(model, pos_reads, neg_reads, batch_size,
               cut, length, channel_indices, device, norm,
               feature_window_size=DEFAULT_FEATURE_WINDOW_SIZE, log_fh=None):
    """Evaluate on test set. Returns dict of metrics."""
    model.eval()

    def _eval_class(reads, label):
        tp, fp, bc = 0, 0, 0
        accepted, rejected = 0, 0
        buf = []
        with torch.no_grad():
            t0 = time.time()
            for rd in reads:
                if len(rd) < cut + length:
                    rejected += 1
                    continue
                accepted += 1
                buf.append(rd[cut:cut + length])
                if len(buf) >= batch_size:
                    bc += 1
                    feat = select_channels(
                        add_features(buf, norm=norm, s_len=length,
                                     w_len=feature_window_size),
                        channel_indices)
                    pred = model(feat.float().to(device)).max(1).indices
                    tp += (pred == label).sum().item()
                    fp += (pred != label).sum().item()
                    buf = []
            if buf:
                bc += 1
                feat = select_channels(
                    add_features(buf, norm=norm, s_len=length,
                                 w_len=feature_window_size),
                    channel_indices)
                pred = model(feat.float().to(device)).max(1).indices
                tp += (pred == label).sum().item()
                fp += (pred != label).sum().item()
            elapsed = time.time() - t0
        return tp, fp, elapsed / max(bc, 1), accepted, rejected

    tp, fn, pt, pos_acc, pos_rej = _eval_class(pos_reads, 1)
    tn, fp, nt, neg_acc, neg_rej = _eval_class(neg_reads, 0)
    
    if log_fh:
        myprint('accepted pos reads: {}, rejected pos reads: {}, TP: {}, FN: {}'.format(
            pos_acc, pos_rej, tp, fn), log_fh)
        myprint('accepted neg reads: {}, rejected neg reads: {}, TN: {}, FP: {}'.format(
            neg_acc, neg_rej, tn, fp), log_fh)

    total = tp + tn + fp + fn
    acc  = round((tp + tn) * 100 / total, 2) if total else 0
    prec = round(tp * 100 / (tp + fp), 2) if (tp + fp) else 0
    rec  = round(tp * 100 / (tp + fn), 2) if (tp + fn) else 0
    f1   = round(2 * prec * rec / (prec + rec), 2) if (prec + rec) else 0
    return dict(accuracy=acc, precision=prec, recall=rec, f1_score=f1,
                avg_infer_time=round((pt + nt) / 2, 4),
                tp=tp, tn=tn, fp=fp, fn=fn)


# ── Visualisation ────────────────────────────────────────────────────

def plot_ablation(results, output_dir):
    """Bar charts comparing each ablation group against the baseline."""
    baseline = next((r for r in results if r["group"] == "baseline"), None)
    if baseline is None:
        return

    groups = {}
    for r in results:
        if r["group"] == "baseline":
            continue
        groups.setdefault(r["group"], []).append(r)

    metrics = ["accuracy", "precision", "recall", "f1_score"]

    for gname, glist in groups.items():
        rows = [baseline] + glist
        names = [r["name"] for r in rows]
        fig, axes = plt.subplots(1, len(metrics),
                                 figsize=(5 * len(metrics), 5))
        for ax, m in zip(axes, metrics):
            vals = [r.get(m, 0) for r in rows]
            colors = ["#2196F3"] + ["#FF9800"] * len(glist)
            bars = ax.bar(range(len(vals)), vals, color=colors)
            ax.set_xticks(range(len(vals)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel(m)
            ax.set_title(m.replace("_", " ").title())
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{v}", ha="center", va="bottom", fontsize=8)
        plt.suptitle(f"Ablation: {gname}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ablation_{gname}.pdf"),
                    dpi=150)
        plt.close()


def print_summary(results, fh):
    """Print a formatted table to stdout and *fh*."""
    hdr = (f"{'Experiment':<25} {'Ch':<5} {'FWin':<6} {'SE':<6} {'1stBN':<6} {'Drop':<6} "
           f"{'Acc%':<8} {'Prec%':<8} {'Rec%':<8} {'F1%':<8} "
           f"{'InferT':<10} {'TrainT':<10}")
    sep = "-" * len(hdr)
    for line in (sep, hdr, sep):
        print(line); fh.write(line + "\n")
    for r in results:
        line = (f"{r['name']:<25} {len(r['channels']):<5} "
                f"{r.get('feature_window_size', DEFAULT_FEATURE_WINDOW_SIZE):<6} "
                f"{str(r['use_SEBlock']):<6} {str(r.get('use_first_bn', True)):<6} {r['dropout']:<6.1f} "
                f"{r.get('accuracy','-'):<8} {r.get('precision','-'):<8} "
                f"{r.get('recall','-'):<8} {r.get('f1_score','-'):<8} "
                f"{r.get('avg_infer_time','-'):<10} "
                f"{r.get('train_time','-'):<10}")
        print(line); fh.write(line + "\n")
    print(sep); fh.write(sep + "\n")


# ── Single-experiment runner ─────────────────────────────────────────

def run_experiment(exp, args, device):
    """Train + test one ablation configuration. Returns updated *exp*."""
    exp_dir = os.path.join(args.output, exp["name"])
    os.makedirs(exp_dir, exist_ok=True)
    
    exp_log_path = os.path.join(exp_dir, "train_test_log.txt")
    exp_log = open(exp_log_path, "w")

    ch_idx = exp["channels"]
    n_ch = len(ch_idx)
    feature_window_size = exp.get("feature_window_size", DEFAULT_FEATURE_WINDOW_SIZE)

    # Model
    model = CAFES([32, 64, 128, 256, 512], n_fc_neurons=2048, depth=29,
                  shortcut=True, dropout_rate=exp["dropout"],
                  use_SEBlock=exp["use_SEBlock"],
                  n_input_channels=n_ch,
                  use_first_bn=exp.get("use_first_bn", True))
    model = nn.DataParallel(model).to(device)

    # Data loaders
    dl_kw = dict(batch_size=args.batch_size // 2,
                 pin_memory=True, num_workers=args.num_workers)

    pos_train = ChannelSelectDataset(
        LazyTrainDataset(os.path.join(args.train_pos_data_folder, "train.npy"),
                         data_type="pos", norm=exp["norm"],
                         cut=args.cut, length=args.length,
                         feature_window_size=feature_window_size), ch_idx)
    neg_train = ChannelSelectDataset(
        LazyTrainDataset(os.path.join(args.train_neg_data_folder, "train.npy"),
                         data_type="neg", norm=exp["norm"],
                         cut=args.cut, length=args.length,
                         feature_window_size=feature_window_size), ch_idx)

    pos_val_raw = np.load(os.path.join(args.train_pos_data_folder, "valid.npy"),
                          allow_pickle=True)
    neg_val_raw = np.load(os.path.join(args.train_neg_data_folder, "valid.npy"),
                          allow_pickle=True)
    pos_val = select_channels(add_features(
        valid_non_normalization(pos_val_raw, args.cut, args.length,
                                False, 299, 10, 16),
        norm=exp["norm"], s_len=args.length,
        w_len=feature_window_size), ch_idx)
    neg_val = select_channels(add_features(
        valid_non_normalization(neg_val_raw, args.cut, args.length,
                                False, 299, 10, 16),
        norm=exp["norm"], s_len=args.length,
        w_len=feature_window_size), ch_idx)

    pos_train_loader = DataLoader(pos_train, **dl_kw)
    neg_train_loader = DataLoader(neg_train, **dl_kw)
    pos_val_loader = DataLoader(Dataset(pos_val, "pos"), **dl_kw)
    neg_val_loader = DataLoader(Dataset(neg_val, "neg"), **dl_kw)

    model_path = os.path.join(exp_dir, "model.pth")

    # Train
    if os.path.exists(model_path):
        print(f"  [Skip] Checkpoint found. Skipping training for {exp['name']}...")
        # 如果跳过训练，为了防止后续给 exp 字典赋值时报错，需要预设一些默认或占位指标
        best_acc = "Skipped"
        best_loss = "Skipped"
        train_s = 0.0
        early = False
    else:
        print(f"  Training {exp['name']} ...")
        # 你的正常训练代码
        best_acc, best_loss, train_s, early = train_model(
            model, pos_train_loader, neg_train_loader,
            pos_val_loader, neg_val_loader,
            exp_dir, device,
            lr=args.learning_rate, epochs=args.epochs,
            tolerance=args.tolerance
        )

    # Test
    myprint(f"--- Testing {exp['name']} ---", exp_log)
    model.load_state_dict(torch.load(model_path))
    pos_test = np.load(os.path.join(args.test_pos_data_folder, "test.npy"),
                       allow_pickle=True)
    neg_test = np.load(os.path.join(args.test_neg_data_folder, "test.npy"),
                       allow_pickle=True)
    metrics = test_model(model, pos_test, neg_test, args.batch_size,
                         args.cut, args.length, ch_idx, device, exp["norm"],
                         feature_window_size=feature_window_size,
                         log_fh=exp_log)

    exp.update(metrics)
    exp["train_time"] = round(train_s, 2)
    exp["early_stop"] = early
    exp["best_val_acc"] = best_acc
    exp["best_val_loss"] = best_loss
    
    exp_log.close()
    return exp


# ── CLI entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(3407)

    ap = argparse.ArgumentParser(description="CAFES Ablation Study")
    ap.add_argument("--train_pos_data_folder", "-trp", type=str, required=True,
                    help="Training positive dataset folder (train/valid .npy)")
    ap.add_argument("--train_neg_data_folder", "-trn", type=str, required=True,
                    help="Training negative dataset folder (train/valid .npy)")
    ap.add_argument("--test_pos_data_folder", "-tep", type=str, required=True,
                    help="Testing positive dataset folder (test .npy)")
    ap.add_argument("--test_neg_data_folder", "-ten", type=str, required=True,
                    help="Testing negative dataset folder (test .npy)")
    ap.add_argument("--output", "-o", type=str, required=True,
                    help="Root output directory for all experiments")
    ap.add_argument("--cut", "-c", type=int, default=1500)
    ap.add_argument("--length", "-l", type=int, default=3000)
    ap.add_argument("--window_sizes", "-ws", type=feature_window_size_type,
                    nargs="+", default=[DEFAULT_FEATURE_WINDOW_SIZE],
                    help="Feature embedding window sizes to evaluate, default 3")
    ap.add_argument("--batch_size", "-b", type=int, default=1024)
    ap.add_argument("--epochs", "-e", type=int, default=300)
    ap.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    ap.add_argument("--tolerance", "-t", type=int, default=10)
    ap.add_argument("--num_workers", "-nw", type=int, default=0)
    ap.add_argument("--gpu_ids", "-g", type=str, default=None,
                    help="Visible CUDA devices, e.g. '0' or '0,1'")
    ap.add_argument("--experiments", "-exp", type=str, nargs="+",
                    default=None,
                    help="Run only these experiments (by name)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    experiments = get_experiments(args.window_sizes)
    if args.experiments:
        experiments = [e for e in experiments
                       if e["name"] in args.experiments]
        print(f"Running {len(experiments)} selected experiment(s)")
    else:
        print(f"Running all {len(experiments)} experiments")

    log = open(os.path.join(args.output, "ablation_summary.txt"), "w")
    results = []

    for idx, exp in enumerate(experiments):
        print(f"\n{'=' * 60}")
        print(f"[{idx + 1}/{len(experiments)}] {exp['name']}")
        print(f"  Channels : {[CHANNEL_NAMES[c] for c in exp['channels']]}")
        print(f"  SEBlock  : {exp['use_SEBlock']}")
        print(f"  Dropout  : {exp['dropout']}")
        print(f"  Norm     : {exp['norm']}")
        print(f"  FeatWin  : {exp['feature_window_size']}")
        print("=" * 60)

        exp = run_experiment(exp, args, device)
        results.append(exp)

        # Incremental CSV save
        csv_path = os.path.join(args.output, "ablation_results.csv")
        fields = ["name", "group", "channel_desc", "channels",
                  "feature_window_size",
                  "use_SEBlock", "use_first_bn", "dropout", "norm",
                  "accuracy", "precision", "recall", "f1_score",
                  "avg_infer_time", "train_time", "early_stop",
                  "best_val_acc", "best_val_loss",
                  "tp", "tn", "fp", "fn"]
        with open(csv_path, "w", newline="") as cf:
            w = csv.DictWriter(cf, fieldnames=fields,
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(results)

    # Final summary
    print(f"\n{'=' * 60}")
    print("ABLATION STUDY COMPLETE")
    print("=" * 60)
    print_summary(results, log)
    plot_ablation(results, args.output)
    log.close()
    print(f"\nResults saved to {args.output}/")
