import argparse
import os
import random
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from nanodeep.npy_signal_utils import (
    NpySignalDataset,
    evaluate_model,
    import_model_class,
    load_model_args,
    resolve_device,
    save_label_map,
    write_eval_outputs,
    write_history_csv,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_count = 0
    correct_count = 0

    pbar = tqdm(data_loader, desc="Train", leave=False)
    for signals, targets in pbar:
        signals = signals.to(device)
        targets = targets.to(device)

        logits = model(signals)
        loss = criterion(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        batch_size = signals.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_count += batch_size
        correct_count += int((predictions == targets).sum().item())
        pbar.set_postfix(
            loss="{:.4f}".format(total_loss / total_count),
            acc="{:.4f}".format(correct_count / total_count),
        )

    return total_loss / total_count, correct_count / total_count


def build_optimizer(name, model, lr, momentum):
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    raise ValueError("unknown optimizer: {}".format(name))


def write_label_order(label_names):
    for index, label_name in enumerate(label_names):
        print("Label_{}: {}".format(index, label_name))


def write_training_summary(history, save_path):
    csv_path = Path(save_path) / "training_history.csv"
    write_history_csv(history, csv_path)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    valid_loss = [row["valid_loss"] for row in history]
    train_acc = [row["train_acc"] for row in history]
    valid_acc = [row["valid_acc"] for row in history]

    plt.figure(figsize=[7, 7], dpi=500)
    plt.plot(epochs, train_loss, label="Train loss", color="b", lw=3)
    plt.plot(epochs, valid_loss, label="Validation loss", color="r", lw=3)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("loss", fontsize=16)
    plt.legend(loc="upper right")
    plt.savefig(Path(save_path) / "loss.png")
    plt.close()

    plt.figure(figsize=[7, 7], dpi=500)
    plt.plot(epochs, train_acc, label="Train accuracy", color="b", lw=3)
    plt.plot(epochs, valid_acc, label="Validation accuracy", color="r", lw=3)
    plt.xlabel("epoch", fontsize=16)
    plt.ylabel("accuracy", fontsize=16)
    plt.legend(loc="lower right")
    plt.savefig(Path(save_path) / "metrics.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Train a NanoDeep model from per-species train/valid/test npy files"
    )
    parser.add_argument("-data_path", required=True, type=str, help="Directory with one subdirectory per species")
    parser.add_argument("-save_path", required=True, type=str, help="Directory used to save the trained model and metrics")
    parser.add_argument("-model_name", default="nanodeep", type=str, help="Model module name under read_deep/model")
    parser.add_argument("-model_config", default=None, type=str, help="Optional model yaml config")
    parser.add_argument("-species", nargs="+", default=None, help="Species directory names to use, in label order")
    parser.add_argument("-signal_length", default=None, type=int, help="Crop or pad each signal to this length; default is npy L")
    parser.add_argument("-epochs", default=30, type=int, help="Number of training epochs")
    parser.add_argument("-batch_size", default=50, type=int, help="Batch size")
    parser.add_argument("-device", default="cuda:0", type=str, help="Device, for example cuda:0 or cpu")
    parser.add_argument("-lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("-momentum", default=0.9, type=float, help="SGD momentum")
    parser.add_argument("-optimizer", default="sgd", choices=("sgd", "adam"), help="Optimizer")
    parser.add_argument("-workers", default=0, type=int, help="DataLoader worker count")
    parser.add_argument("-seed", default=1, type=int, help="Random seed")
    parser.add_argument(
        "-normalize",
        default="nanodeep",
        choices=("zscore", "nanodeep", "none"),
        help="Per-signal normalization mode",
    )
    parser.add_argument("--load_to_mem", default=False, action="store_true", help="Load all npy arrays into memory")
    parser.add_argument("--save_best", default=False, action="store_true", help="Save best validation model only")
    parser.add_argument("--test_model", default=False, action="store_true", help="Evaluate test.npy after training")
    opt = parser.parse_args()

    set_seed(opt.seed)
    os.makedirs(opt.save_path, exist_ok=True)
    device = resolve_device(opt.device)

    train_dataset = NpySignalDataset(
        data_path=opt.data_path,
        split="train",
        signal_length=opt.signal_length,
        species_order=opt.species,
        normalize=opt.normalize,
        mmap=not opt.load_to_mem,
    )
    valid_dataset = NpySignalDataset(
        data_path=opt.data_path,
        split="valid",
        signal_length=train_dataset.signal_length,
        species_order=train_dataset.species_names,
        normalize=opt.normalize,
        mmap=not opt.load_to_mem,
    )

    label_names = train_dataset.species_names
    class_num = len(label_names)
    write_label_order(label_names)
    save_label_map(label_names, Path(opt.save_path) / "labels.json")

    model_args = load_model_args(
        opt.model_name,
        opt.model_config,
        class_num=class_num,
        signal_length=train_dataset.signal_length,
    )
    deepmodel = import_model_class(opt.model_name)
    model = deepmodel(**model_args).to(device)

    print("-data_path:", opt.data_path)
    print("-save_path:", opt.save_path)
    print("-model_name:", opt.model_name)
    print("-device:", device)
    print("-signal_length:", train_dataset.signal_length)
    print("-class_num:", class_num)
    print("-train_samples:", len(train_dataset))
    print("-valid_samples:", len(valid_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.workers,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(opt.optimizer, model, opt.lr, opt.momentum)
    model_save_path = Path(opt.save_path) / "model.pth"
    best_valid_acc = -1.0
    history = []

    for epoch in range(1, opt.epochs + 1):
        print("Epoch {}/{}".format(epoch, opt.epochs))
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        valid_metrics = evaluate_model(model, valid_loader, device, class_num, criterion)
        valid_loss = valid_metrics["loss"]
        valid_acc = valid_metrics["accuracy"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "valid_loss": valid_loss,
                "valid_acc": valid_acc,
            }
        )
        print(
            "train_loss={:.4f} train_acc={:.4f} valid_loss={:.4f} valid_acc={:.4f}".format(
                train_loss, train_acc, valid_loss, valid_acc
            )
        )

        if opt.save_best:
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), model_save_path)
                print("saved best model:", model_save_path)

    if not opt.save_best:
        torch.save(model.state_dict(), model_save_path)
        print("saved model:", model_save_path)

    write_training_summary(history, opt.save_path)
    write_eval_outputs(valid_metrics, label_names, Path(opt.save_path) / "validation")

    if opt.test_model:
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        test_dataset = NpySignalDataset(
            data_path=opt.data_path,
            split="test",
            signal_length=train_dataset.signal_length,
            species_order=label_names,
            normalize=opt.normalize,
            mmap=not opt.load_to_mem,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.workers,
        )
        test_metrics = evaluate_model(model, test_loader, device, class_num, criterion)
        write_eval_outputs(test_metrics, label_names, Path(opt.save_path) / "test")
        print("test_acc={:.4f} test_loss={:.4f}".format(test_metrics["accuracy"], test_metrics["loss"]))


if __name__ == "__main__":
    main()
