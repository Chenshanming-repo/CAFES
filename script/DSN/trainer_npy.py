"""Train a DeepSelectNet binary classifier from NxL npy signal files.

Data layout (per-class directories, mirroring NanoDeep):

    data_path/
        <target_name>/{train,valid,test}.npy
        <nontarget_name>/{train,valid,test}.npy

Each npy holds N variable-length reads (shape NxL, Nx1xL, or an object array
of 1D reads). Reads are normalized and cropped/padded to -signal_length.
"""

import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from timeit import default_timer as timer

path = str(Path(Path(__file__).parent.absolute()).parent.absolute())
sys.path.insert(0, path)

import click
import numpy as np
import tensorflow as tf

from core.FCN import FCN
from core.ResNet import RESNET
from core.InceptionNet import InceptionNet
from core.TransformerNet import TransformerNet
from core.Logger import Logger
from core.npy_signal_utils import (
    NpyBinarySignalSequence,
    evaluate_predictions,
    save_label_map,
    write_eval_outputs,
    write_history_csv,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def get_classifier(classifier, in_shape, nb_classes=1, is_train=False):
    if classifier == "FCN":
        model = FCN(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif classifier == "ResNet":
        model = RESNET(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif classifier == "InceptionNet":
        model = InceptionNet(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    elif classifier == "TransformerNet":
        model = TransformerNet(input_shape=in_shape, nb_classes=nb_classes, is_train=is_train).model
    else:
        raise ValueError("unknown classifier: {}".format(classifier))

    model.compile(
        loss=tf.keras.metrics.BinaryCrossentropy().name,
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall(),
        ],
    )
    return model


def draw_history(history, output_path):
    try:
        import matplotlib

        matplotlib.use("agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    epochs = [row["epoch"] for row in history]
    plt.figure(0)
    plt.title("Train vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, [row["train_loss"] for row in history], label="train")
    plt.plot(epochs, [row["valid_loss"] for row in history], label="validation")
    plt.legend()
    plt.savefig(str(Path(output_path) / "loss.png"))
    plt.close(0)

    plt.figure(1)
    plt.title("Train vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epochs, [row["train_acc"] for row in history], label="train")
    plt.plot(epochs, [row["valid_acc"] for row in history], label="validation")
    plt.legend()
    plt.savefig(str(Path(output_path) / "accuracy.png"))
    plt.close(1)


@click.command()
@click.option("--data_path", "-d", required=True, type=click.Path(exists=True),
              help="Directory with one subdirectory per class")
@click.option("--target", "-t", required=True, type=str,
              help="Target class directory name (label 1)")
@click.option("--nontarget", "-nt", required=True, type=str,
              help="Non-target class directory name (label 0)")
@click.option("--output", "-o", required=True, type=click.Path(exists=False),
              help="Output directory for the trained model and metrics")
@click.option("--classifier", "-c", default="ResNet",
              type=click.Choice(["FCN", "ResNet", "InceptionNet", "TransformerNet"]),
              help="Model architecture, default=ResNet")
@click.option("--signal_length", "-sl", default=3000, type=int,
              help="Crop/pad each read to this length, default=3000")
@click.option("--epochs", "-e", default=100, type=int, help="Number of epochs")
@click.option("--batch", "-b", default=200, type=int, help="Training batch size")
@click.option("--test_batch", "-tb", default=50, type=int, help="Evaluation batch size")
@click.option("--lr", "-lr", default=0.001, type=float, help="Adam learning rate")
@click.option("--normalize", "-n", default="deepselectnet",
              type=click.Choice(["deepselectnet", "zscore", "none"]),
              help="Per-signal normalization, default=deepselectnet")
@click.option("--seed", "-s", default=1, type=int, help="Random seed")
@click.option("--load_to_mem", "-mem", default=False, is_flag=True,
              help="Load npy arrays fully into memory instead of memory-mapping")
@click.option("--test_model", "-test", default=False, is_flag=True,
              help="Evaluate test.npy after training")
def main(data_path, target, nontarget, output, classifier, signal_length, epochs,
         batch, test_batch, lr, normalize, seed, load_to_mem, test_model):
    set_seed(seed)
    os.makedirs(output, exist_ok=True)
    sys.stdout = Logger(str(Path(output) / "train-log.txt"))

    mmap = not load_to_mem

    train_seq = NpyBinarySignalSequence(
        data_path=data_path, split="train", target_name=target, nontarget_name=nontarget,
        signal_length=signal_length, batch_size=batch, normalize=normalize,
        mmap=mmap, shuffle=True, seed=seed,
    )
    signal_length = train_seq.signal_length
    valid_seq = NpyBinarySignalSequence(
        data_path=data_path, split="valid", target_name=target, nontarget_name=nontarget,
        signal_length=signal_length, batch_size=test_batch, normalize=normalize,
        mmap=mmap, shuffle=False, seed=seed,
    )

    save_label_map(target, nontarget, Path(output) / "labels.json")

    in_shape = (signal_length, 1)
    model = get_classifier(classifier, in_shape, nb_classes=1, is_train=False)
    model.optimizer.learning_rate = lr

    print("****Train Configs****")
    print("Classifier: ", classifier)
    print("Data path: ", data_path)
    print("Target (label 1): ", target)
    print("Non-target (label 0): ", nontarget)
    print("Input shape: ", in_shape)
    print("Signal length: ", signal_length)
    print("Epochs: ", epochs)
    print("Train batch size: ", batch)
    print("Normalization: ", normalize)
    print("Learning rate: ", lr)
    print("Train samples: ", len(train_seq.index))
    print("Valid samples: ", len(valid_seq.index))

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_binary_accuracy", factor=0.5, patience=20, mode="max",
        min_lr=0, verbose=1,
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_binary_accuracy", patience=30, verbose=1,
        restore_best_weights=True,
    )
    model_path = str(Path(output) / "model")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor="val_binary_accuracy",
        save_best_only=True, verbose=1,
    )

    start = timer()
    fit_history = model.fit(
        train_seq,
        validation_data=valid_seq,
        epochs=epochs,
        callbacks=[reduce_lr, early_stopping, checkpoint],
        verbose=1,
    )
    end = timer()
    print("Time Elapsed: {}".format(str(timedelta(seconds=end - start))))

    history = []
    ran_epochs = len(fit_history.history["loss"])
    for i in range(ran_epochs):
        history.append({
            "epoch": i + 1,
            "train_loss": fit_history.history["loss"][i],
            "train_acc": fit_history.history["binary_accuracy"][i],
            "valid_loss": fit_history.history["val_loss"][i],
            "valid_acc": fit_history.history["val_binary_accuracy"][i],
        })
    write_history_csv(history, str(Path(output) / "training_history.csv"))
    draw_history(history, output)

    model.save(model_path)
    print("Saved model to ", model_path)

    valid_metrics = evaluate_predictions(model, valid_seq, test_batch)
    write_eval_outputs(valid_metrics, target, nontarget, Path(output) / "validation")
    print("valid_acc={:.4f} valid_f1={:.4f}".format(
        valid_metrics["accuracy"], valid_metrics["f1"]))

    if test_model:
        test_seq = NpyBinarySignalSequence(
            data_path=data_path, split="test", target_name=target, nontarget_name=nontarget,
            signal_length=signal_length, batch_size=test_batch, normalize=normalize,
            mmap=mmap, shuffle=False, seed=seed,
        )
        test_metrics = evaluate_predictions(model, test_seq, test_batch)
        write_eval_outputs(test_metrics, target, nontarget, Path(output) / "test")
        print("test_acc={:.4f} test_f1={:.4f}".format(
            test_metrics["accuracy"], test_metrics["f1"]))

    sys.stdout.close()


if __name__ == "__main__":
    main()
