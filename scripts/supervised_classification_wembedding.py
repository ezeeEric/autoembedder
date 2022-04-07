"""
If a model is provided using a second argument, the classification is run using the pretrained autoembedder.

- python scripts/supervised_classification_wembedding.py ./data/training_input/ [./data/model/autoembedder]
"""

import sys
import pandas as pd
import tensorflow as tf

from utils.params import with_params
from utils.utils import get_sorted_input_files, load_model
from utils.engine import compile_model
from utils.data import prepare_data
from utils.visualisation import plot_metrics_history, write_metrics

from models.base_classification_network import BaseClassificationNetwork
from models.autoembedder_classification_model import AutoEmbedderClassificationModel


OUTPUT_DIRECTORY = ""


def train_model(
    train_data_num: pd.DataFrame,
    train_data_cat: pd.DataFrame,
    train_data_target: pd.DataFrame,
    model: tf.keras.Model,
    config: dict,
) -> tf.keras.callbacks.History:

    history = model.fit(
        [train_data_cat, train_data_num],
        train_data_target,
        validation_split=config["val_data_fraction"],
        batch_size=config["batch_size"],
        epochs=config["n_epochs"],
        verbose=config["verbosity_level"],
    )
    return history


def test_model(
    test_data_num: pd.DataFrame,
    test_data_cat: pd.DataFrame,
    test_data_target: pd.DataFrame,
    model: tf.keras.Model,
    config: dict,
) -> None:
    loss, accuracy = model.evaluate(
        [test_data_cat, test_data_num],
        test_data_target,
        batch_size=config["batch_size"],
        verbose=config["verbosity_level"],
    )
    print(f" Model loss on the test set: {loss:.2E}")
    print(f" Model accuracy on the test set: {100*accuracy:.1f}%")
    return loss, accuracy


@with_params("params.yaml", "train_classification_models")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns="",
        input_extension="feather",
    )

    df = pd.read_feather(input_files[0])

    (
        train_df_num,
        train_df_cat,
        test_df_num,
        test_df_cat,
        train_df_target,
        test_df_target,
        encoding_reference_values,
        _,
    ) = prepare_data(df, params)

    n_target_classes = train_df_target.shape[1]

    if len(sys.argv) > 2:
        print(f"Pretrained Autoembedder {sys.argv[2]} will be used in classification.")
        autoembedder = load_model(sys.argv[2])
        model = AutoEmbedderClassificationModel(
            n_numerical_inputs=len(train_df_num.columns),
            n_target_classes=n_target_classes,
            autoembedder=autoembedder,
            encoding_reference_values=encoding_reference_values,
            config=params,
        )
    else:
        print(f"No pretrained model defined, using basic model.")
        model = BaseClassificationNetwork(
            n_numerical_inputs=len(train_df_num.columns),
            n_target_classes=n_target_classes,
            encoding_reference_values=encoding_reference_values,
            config=params,
        )
    compile_model(model=model, config=params)
    history = train_model(
        train_data_num=train_df_num,
        train_data_cat=train_df_cat,
        train_data_target=train_df_target,
        model=model,
        config=params,
    )
    plot_metrics_history(history=history, outdir="./data/plots/", tag="autoembedder")
    loss, accuracy = test_model(
        test_data_num=test_df_num,
        test_data_cat=test_df_cat,
        test_data_target=test_df_target,
        model=model,
        config=params,
    )
    write_metrics(loss, accuracy, outdir="./data/metrics/")


if __name__ == "__main__":
    main()
