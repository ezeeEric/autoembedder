"""
Script to train the Autoembedder model. Input is a single file containing the
full vocabulary. Feature selection follows the data preprocessing. Output is the
trained model.

python scripts/train_autoembedder.py ./data/training_input <input_pattern>
"""

import sys
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from utils.params import with_params
from utils.utils import get_sorted_input_files, save_model

from scripts.preprocess_data import load_features

from autoembedder.autoembedder import AutoEmbedder, AutoembedderCallbacks


def create_encoding_reference_values(
    feature_names: list,
    encoder: OrdinalEncoder,
) -> dict[str, list]:
    """Convert a np.ndarray without column names to a dictionary of lists. Key
    is the feature name, values are the different categories used during
    encoding."""
    return {
        feature_name: encoder.categories_[i]
        for i, feature_name in enumerate(feature_names)
    }


# TODO check duplication with apply_...
def encode_categorical_input_ordinal(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, dict]:

    embedding_input_encoder = OrdinalEncoder()
    data_enc = embedding_input_encoder.fit_transform(df)

    df_enc = pd.DataFrame(data_enc, columns=df.columns)

    encoding_reference_values = create_encoding_reference_values(
        df.columns, embedding_input_encoder
    )

    return df_enc, encoding_reference_values


# TODO should this be in preprocessing?
def normalise_numerical_input_columns(
    df: pd.DataFrame, method: str = "minmax"
) -> pd.DataFrame:
    if method == "minmax":
        df_transformed = pd.DataFrame(
            MinMaxScaler().fit_transform(df), columns=df.columns
        )
    elif method == "standard":
        df_transformed = pd.DataFrame(
            StandardScaler().fit_transform(df), columns=df.columns
        )
    elif method == "manual":
        epsilon = 1e-12
        df_transformed = (df - df.min()) / (df.max() - df.min() + epsilon)
    else:
        raise NotImplementedError(f"{method} not a valid transformation method.")
    return df_transformed


# TODO should this be split up a bit?
def compile_model(
    model: tf.keras.Model,
    learning_rate: float,
    optimizer_name: str = "sgd",
    loss_name: str = "mse",
    metrics: list[str] = [],
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

    selected_metrics = []

    for metric_name in list(metrics):
        if metric_name == "accuracy":
            selected_metrics.append(tf.keras.metrics.Accuracy())
        elif metric_name == "precision":
            selected_metrics.append(tf.keras.metrics.Precision())
        else:
            raise NotImplementedError(f"Metric {metric_name} not implemented.")

    # explicitely setting run_eagerly=True is necessary in tf 2.0 when dealing
    # with custom layers and losses
    model.compile(
        optimizer=optimizer, loss=loss, metrics=selected_metrics, run_eagerly=True
    )


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
    model.create_embedding_reference()
    model.evaluate(tf.convert_to_tensor(df), batch_size=batch_size)


def prepare_data_for_fit(
    df: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    normalisation_method: str,
    test_data_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """This function first encodes the categorical input, then normalises the numerical input and finally merges the result."""
    # TODO this is somewhat duplicated with apply_ordinal_encoding_column()
    df_encoded, embedding_encoder = encode_categorical_input_ordinal(
        df[categorical_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(
        df[numerical_features], method=normalisation_method
    )
    df = pd.concat([df_numericals_normalised, df_encoded], axis=1)
    train_df, test_df = train_test_split(df, test_size=test_data_fraction)
    return train_df, test_df, embedding_encoder


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
