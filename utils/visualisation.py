import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def plot_loss_history(history: tf.keras.callbacks.History, outdir: str) -> None:
    plt.plot(history.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train"], loc="upper left")
    target_dir = os.path.join(".", outdir, "")
    os.makedirs(target_dir, exist_ok=True)
    plt.savefig(f"{outdir}/loss.pdf")


def write_metrics(loss: float, accuracy: float, outdir: str) -> None:
    target_dir = os.path.join(".", outdir, "")
    os.makedirs(target_dir, exist_ok=True)
    with open(f"{target_dir}/metrics.txt", "w") as outfile:
        outfile.write(f"Loss on the test set: {loss:.2E}\n")
        outfile.write(f"Accuracy on the test set: {100*accuracy:.1f}%")
