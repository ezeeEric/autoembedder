"""
Script to train the Autoembedder model. Input is a single file containing the
full vocabulary. Feature selection follows the data preprocessing. Output is the
trained model.

python scripts/train_autoembedder.py ./data/training_input <input_pattern>
"""

import sys
import pandas as pd
import tensorflow as tf

import utils.engine as engine
from utils.params import with_params
from utils.utils import (
    get_sorted_input_files,
    save_model,
)
from utils.data import (
    load_features,
    encode_categorical_input_ordinal,
    normalise_numerical_input_columns,
)
from utils.visualisation import plot_metrics_history, write_metrics

from sklearn.model_selection import train_test_split

from autoembedder.autoembedder import AutoEmbedder, AutoembedderCallbacks


def train_model(
    df: pd.DataFrame, validation_df: pd.DataFrame, model: AutoEmbedder, config: dict
) -> tf.keras.callbacks.History:
    autoembedder_callback = AutoembedderCallbacks(
        validation_data=tf.convert_to_tensor(validation_df)
    )
    model.match_feature_to_input_column_idx(columns=df.columns)
    history = model.fit(
        tf.convert_to_tensor(df),
        batch_size=config["batch_size"],
        validation_split=config["val_data_fraction"],
        epochs=config["n_epochs"],
        verbose=1,
        callbacks=[autoembedder_callback],
    )
    history.history = history.history | autoembedder_callback.metric_history
    return history


def test_model(df: pd.DataFrame, model: AutoEmbedder, batch_size: int) -> None:
    autoembedder_callback = AutoembedderCallbacks(
        validation_data=tf.convert_to_tensor(df)
    )
    metrics = model.evaluate(
        tf.convert_to_tensor(df),
        batch_size=batch_size,
        callbacks=[autoembedder_callback],
    )
    print(metrics)
    exit()


def train_autoembedder(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    numerical_features, categorical_features, _ = load_features(
        df, params["feature_handler_file"]
    )
    df_encoded, encoding_reference_values = encode_categorical_input_ordinal(
        df[categorical_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(
        df[numerical_features], method=params["normalisation_method"]
    )
    df = pd.concat([df_numericals_normalised, df_encoded], axis=1)

    train_df, test_df = train_test_split(df, test_size=params["test_data_fraction"])

    auto_embedder = AutoEmbedder(
        encoding_reference_values=encoding_reference_values,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        config=params,
    )
    engine.compile_model(model=auto_embedder, config=params)
    history = train_model(
        df=train_df, validation_df=test_df, model=auto_embedder, config=params
    )
    plot_metrics_history(history=history, outdir="./data/plots/")

    test_model(
        test_df,
        auto_embedder,
        batch_size=params["batch_size"],
    )
    write_metrics(loss, accuracy, outdir="./data/metrics/")

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
