import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from nanodeep.npy_signal_utils import (
    NpySignalDataset,
    discover_species,
    evaluate_model,
    import_model_class,
    load_label_map,
    load_model_args,
    resolve_device,
    resolve_labels_path,
    write_eval_outputs,
)


def main():
    parser = argparse.ArgumentParser(
        description="Test a NanoDeep model from per-species npy files"
    )
    parser.add_argument("-data_path", required=True, type=str, help="Directory with one subdirectory per species")
    parser.add_argument("-save_path", required=True, type=str, help="Directory used to save test metrics")
    parser.add_argument("-model_path", required=True, type=str, help="Model .pth file to load")
    parser.add_argument("-model_name", default="nanodeep", type=str, help="Model module name under read_deep/model")
    parser.add_argument("-model_config", default=None, type=str, help="Optional model yaml config")
    parser.add_argument("-labels_path", default=None, type=str, help="Optional labels.json saved by training")
    parser.add_argument("-species", nargs="+", default=None, help="Species directory names to use if labels.json is not available")
    parser.add_argument("-split", default="test", choices=("train", "valid", "validation", "val", "test"), help="Split to evaluate")
    parser.add_argument("-signal_length", default=None, type=int, help="Crop or pad each signal to this length; default is npy L")
    parser.add_argument("-batch_size", default=50, type=int, help="Batch size")
    parser.add_argument("-device", default="cuda:0", type=str, help="Device, for example cuda:0 or cpu")
    parser.add_argument("-workers", default=0, type=int, help="DataLoader worker count")
    parser.add_argument(
        "-normalize",
        default="nanodeep",
        choices=("zscore", "nanodeep", "none"),
        help="Per-signal normalization mode; use the same value as training",
    )
    parser.add_argument("--load_to_mem", default=False, action="store_true", help="Load all npy arrays into memory")
    opt = parser.parse_args()

    os.makedirs(opt.save_path, exist_ok=True)
    device = resolve_device(opt.device)

    labels_path = resolve_labels_path(opt.model_path, opt.save_path, opt.labels_path)
    if labels_path is not None:
        label_names = load_label_map(labels_path)
    elif opt.species is not None:
        label_names = opt.species
    else:
        label_names = discover_species(opt.data_path, opt.split)

    dataset = NpySignalDataset(
        data_path=opt.data_path,
        split=opt.split,
        signal_length=opt.signal_length,
        species_order=label_names,
        normalize=opt.normalize,
        mmap=not opt.load_to_mem,
    )
    class_num = len(label_names)
    model_args = load_model_args(
        opt.model_name,
        opt.model_config,
        class_num=class_num,
        signal_length=dataset.signal_length,
    )
    deepmodel = import_model_class(opt.model_name)
    model = deepmodel(**model_args).to(device)
    model.load_state_dict(torch.load(opt.model_path, map_location=device))

    print("-data_path:", opt.data_path)
    print("-save_path:", opt.save_path)
    print("-model_path:", opt.model_path)
    print("-model_name:", opt.model_name)
    print("-device:", device)
    print("-split:", opt.split)
    print("-signal_length:", dataset.signal_length)
    print("-class_num:", class_num)
    print("-samples:", len(dataset))

    data_loader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.workers,
    )
    metrics = evaluate_model(model, data_loader, device, class_num)
    write_eval_outputs(metrics, label_names, opt.save_path)
    print("accuracy={:.4f} loss={:.4f}".format(metrics["accuracy"], metrics["loss"]))


if __name__ == "__main__":
    main()
