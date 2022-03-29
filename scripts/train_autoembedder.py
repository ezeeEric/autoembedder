"""
Script to train the Autoembedder model. Input is a single file containing the
full vocabulary. Feature selection follows the data preprocessing. Output is the
trained model.

python scripts/train_autoembedder.py ./data/training_input <input_pattern>
"""

import sys
import pandas as pd
import tensorflow as tf

from utils.params import with_params
from utils.utils import (
    get_sorted_input_files,
    save_model,
    prepare_data_for_fit,
    load_features,
)


from autoembedder.autoembedder import AutoEmbedder, AutoembedderCallbacks


# TODO should this be split up a bit?
def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str = "sgd",
    loss_name: str = "mse",
) -> None:

    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        raise NotImplementedError()

    if loss_name == "mse":
        loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
    elif loss_name == "bce":
        loss = tf.keras.losses.BinaryCrossentropy()
    elif loss_name == "cce":
        loss = tf.keras.losses.CategoricalCrossentropy()
    else:
        raise NotImplementedError(f"Metric {loss_name} not implemented.")

    # explicitely setting run_eagerly=True is necessary in tf 2.0 when dealing
    # with custom layers and losses
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)


def train_model(
    df: pd.DataFrame, model: AutoEmbedder, batch_size: int, epochs: int
) -> None:
    model.match_feature_to_input_column_idx(columns=df.columns)
    model.fit(
        tf.convert_to_tensor(df),
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[AutoembedderCallbacks()],
    )


def test_model(df: pd.DataFrame, model: AutoEmbedder, batch_size: int) -> None:
    model.evaluate(
        tf.convert_to_tensor(df),
        batch_size=batch_size,
        callbacks=[AutoembedderCallbacks()],
    )


def train_autoembedder(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    numerical_features, categorical_features, _ = load_features(
        df, params["feature_handler_file"]
    )
    train_df, test_df, encoding_reference_values = prepare_data_for_fit(
        df,
        numerical_features,
        categorical_features,
        normalisation_method=params["normalisation_method"],
        test_data_fraction=params["test_data_fraction"],
    )
    auto_embedder = AutoEmbedder(
        encoding_reference_values=encoding_reference_values,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        config=params,
    )
    compile_model(
        model=auto_embedder,
        learning_rate=params["learning_rate"],
        optimizer_name=params["optimizer"],
        loss_name=params["loss"],
    )
    train_model(
        df=train_df,
        model=auto_embedder,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
    )
    test_model(
        test_df,
        auto_embedder,
        batch_size=params["batch_size"],
    )
    return auto_embedder


@with_params("params.yaml", "train_autoembedder")
def main(params):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[2].split(",") if len(sys.argv) > 2 else None,
        input_extension="feather",
    )

    if len(input_files) == 1:
        df = pd.read_feather(input_files[0])
    else:
        print(f"Warning, expected single file in input, got: {input_files}")

    autoembedder_model = train_autoembedder(df, params)

    save_model(autoembedder_model, params["output_directory"])


if __name__ == "__main__":
    main()
