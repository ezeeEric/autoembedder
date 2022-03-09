"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder

print(f"tensorflow {tf.__version__}")

from autoembedder.autoembedder import AutoEmbedder
from utils.feature_handler import FeatureHandler
from utils.params import with_params
from scripts.preprocess_data import select_features, create_output_dir

OUTPUT_DIRECTORY = "./data/model"


def create_encoding_dictionary(
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


def encode_categorical_input_ordinal(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, dict]:

    embedding_input_encoder = OrdinalEncoder()
    data_enc = embedding_input_encoder.fit_transform(df)

    df_enc = pd.DataFrame(data_enc, columns=df.columns)

    encoding_dictionary = create_encoding_dictionary(
        df.columns, embedding_input_encoder
    )

    return df_enc, encoding_dictionary


def save_auto_embedding_model(
    auto_embedder: AutoEmbedder,
    out_path: str,
) -> None:
    out_file = os.path.join(out_path, "auto_embedder")
    auto_embedder.save(out_file)


def normalise_numerical_input_columns(df: pd.DataFrame) -> pd.DataFrame:
    # TODO
    # - the epsilon here is required to avoid dividing by 0 TODO the
    # - normalisation on columns with large range yields many values close to 0.5.
    # Should this be scaled differently?
    # mean scale
    # df_mean_normed = (df - df.mean()) / (df.std())
    # minmax scale
    epsilon = 1e-12
    df_normed = (df - df.min()) / (df.max() - df.min() + epsilon)
    return df_normed


def compile_model(
    model: AutoEmbedder,
    learning_rate: float,
    optimizer_name: str = "sgd",
    loss_name: str = "mse",
) -> None:

    if optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise NotImplementedError()

    if loss_name == "mse":
        loss = tf.keras.losses.MeanSquaredError()
    else:
        raise NotImplementedError()

    # explicitely setting run_eagerly=True is necessary in tf 2.0 when dealing
    # with custom layers and losses
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=True)


def train_model(
    df: pd.DataFrame, model: AutoEmbedder, batch_size: int, epochs: int
) -> None:
    model.match_feature_to_input_column_idx(columns=df.columns)
    model.fit(tf.convert_to_tensor(df), batch_size=batch_size, epochs=epochs)
    return model


def test_model(df: pd.DataFrame, model: AutoEmbedder, batch_size: int) -> None:
    # TODO a proper evaluation procedure still needs to be implemented
    # model.evaluate(tf.convert_to_tensor(df), batch_size=batch_size)
    pass


def prepare_data_for_fit(
    df: tf.Tensor,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tf.Tensor:
    """This function first encodes the categorical input, then normalises the numerical input and finally merges the result."""
    df_encoded, embedding_encoder = encode_categorical_input_ordinal(
        df[categorical_features]
    )
    df_numericals_no_nan = report_preprocessing.fill_na_with_median(
        df[numerical_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(df_numericals_no_nan)
    df = pd.concat([df_numericals_normalised, df_encoded], axis=1)
    return df, embedding_encoder


def train_autoembedder(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    feature_handler = FeatureHandler.from_json(params["feature_handler_file"])
    numerical_features, categorical_features = select_features(df, feature_handler)
    df, encoding_dictionary = prepare_data_for_fit(
        df, numerical_features, categorical_features
    )
    auto_embedder = AutoEmbedder(
        encoding_dictionary, numerical_features, categorical_features
    )
    compile_model(
        model=auto_embedder,
        learning_rate=params["learning_rate"],
        optimizer_name=params["optimizer"],
        loss_name=params["loss"],
    )

    auto_embedder = train_model(
        df=df,
        model=auto_embedder,
        batch_size=params["batch_size"],
        epochs=params["epochs"],
    )
    test_model(
        df,
        auto_embedder,
        batch_size=params["batch_size"],
    )
    return auto_embedder


@with_params("params.yaml", "train_model")
def main(params):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[2].split(",") if len(sys.argv) > 2 else None,
        input_extension="feather",
    )
    create_output_dir(OUTPUT_DIRECTORY)

    df = pd.read_feather(input_files[0])

    auto_embedding_model = train_autoembedder(df, params)
    save_auto_embedding_model(auto_embedding_model, OUTPUT_DIRECTORY)


if __name__ == "__main__":
    main()