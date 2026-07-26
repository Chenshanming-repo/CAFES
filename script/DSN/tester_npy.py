"""Test a trained DeepSelectNet model on NxL npy signal files.

Loads a saved Keras model and evaluates it on the target/non-target npy files
of the requested split (default: test). Useful for cross-batch / cross-speed
experiments where the test classes differ from the training classes.
"""

import os
import sys
from pathlib import Path

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import click
import tensorflow as tf
from tensorflow import keras

from core.Logger import Logger
from core.npy_signal_utils import (
    NpyBinarySignalSequence,
    evaluate_predictions,
    load_label_map,
    resolve_labels_path,
    write_eval_outputs,
)


@click.command()
@click.option("--data_path", "-d", required=True, type=click.Path(exists=True),
              help="Directory with one subdirectory per class")
@click.option("--model", "-m", "model_path", required=True, type=click.Path(exists=True),
              help="Saved Keras model directory")
@click.option("--output", "-o", required=True, type=click.Path(exists=False),
              help="Output directory for test metrics")
@click.option("--target", "-t", default=None, type=str,
              help="Target class directory name (label 1); defaults to labels.json")
@click.option("--nontarget", "-nt", default=None, type=str,
              help="Non-target class directory name (label 0); defaults to labels.json")
@click.option("--labels_path", "-lp", default=None, type=click.Path(exists=True),
              help="Optional labels.json saved during training")
@click.option("--split", "-sp", default="test",
              type=click.Choice(["train", "valid", "validation", "val", "test"]),
              help="Split to evaluate, default=test")
@click.option("--signal_length", "-sl", default=3000, type=int,
              help="Crop/pad each read to this length, default=3000")
@click.option("--batch", "-b", default=50, type=int, help="Evaluation batch size")
@click.option("--normalize", "-n", default="deepselectnet",
              type=click.Choice(["deepselectnet", "zscore", "none"]),
              help="Per-signal normalization; use the same value as training")
@click.option("--load_to_mem", "-mem", default=False, is_flag=True,
              help="Load npy arrays fully into memory instead of memory-mapping")
def main(data_path, model_path, output, target, nontarget, labels_path, split,
         signal_length, batch, normalize, load_to_mem):
    os.makedirs(output, exist_ok=True)
    sys.stdout = Logger(str(Path(output) / "test-log.txt"))

    if target is None or nontarget is None:
        resolved = resolve_labels_path(model_path, output, labels_path)
        if resolved is None:
            raise ValueError(
                "target/nontarget not given and no labels.json found; "
                "pass -t and -nt explicitly"
            )
        map_target, map_nontarget = load_label_map(resolved)
        target = target or map_target
        nontarget = nontarget or map_nontarget

    sequence = NpyBinarySignalSequence(
        data_path=data_path, split=split, target_name=target, nontarget_name=nontarget,
        signal_length=signal_length, batch_size=batch, normalize=normalize,
        mmap=not load_to_mem, shuffle=False,
    )

    model = keras.models.load_model(model_path)
    model.compile(
        loss=tf.keras.metrics.BinaryCrossentropy().name,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )

    print("****Test Configs****")
    print("Data path: ", data_path)
    print("Model path: ", model_path)
    print("Target (label 1): ", target)
    print("Non-target (label 0): ", nontarget)
    print("Split: ", split)
    print("Signal length: ", sequence.signal_length)
    print("Samples: ", len(sequence.index))

    metrics = evaluate_predictions(model, sequence, batch)
    write_eval_outputs(metrics, target, nontarget, output)
    print("accuracy={:.4f} precision={:.4f} recall={:.4f} f1={:.4f}".format(
        metrics["accuracy"], metrics["precision"], metrics["recall"], metrics["f1"]))

    sys.stdout.close()


if __name__ == "__main__":
    main()
