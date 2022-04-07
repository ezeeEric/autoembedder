import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from typing import Tuple


def find_metric_pairs(history: tf.keras.callbacks.History) -> dict:
    """Find pairs of metric for train and validation set."""
    metric_dict = {}
    metric_list = history.history.keys()
    for metric in metric_list:
        if metric.startswith("val"):
            continue
        if f"val_{metric}" in metric_list:
            metric_dict[metric] = (
                history.history[metric],
                history.history[f"val_{metric}"],
            )
    return metric_dict


def plot_metrics_history(history: tf.keras.callbacks.History, outdir: str) -> None:
    metric_plot_dict = find_metric_pairs(history)
    for metric_name, train_test_metrics in metric_plot_dict.items():
        plt.plot(train_test_metrics[0])
        plt.plot(train_test_metrics[1])
        plt.title(f"Model {metric_name}")
        plt.ylabel(f"{metric_name}")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        target_dir = os.path.join(".", outdir, "")
        os.makedirs(target_dir, exist_ok=True)
        plt.savefig(f"{outdir}/{metric_name}.pdf")
        plt.clf()


def write_metrics(loss: float, accuracy: float, outdir: str) -> None:
    target_dir = os.path.join(".", outdir, "")
    os.makedirs(target_dir, exist_ok=True)
    with open(f"{target_dir}/metrics.txt", "w") as outfile:
        outfile.write(f"Loss on the test set: {loss:.2E}\n")
        outfile.write(f"Accuracy on the test set: {100*accuracy:.1f}%")
