import json
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler

import tensorflow as tf

from utils.utils import create_output_dir
from utils.feature_handler import FeatureHandler


def load_feature_actions(feature_action_file: str) -> dict:
    print("Loading feature action file.")
    with open(feature_action_file) as f:
        feature_actions = json.load(f)
    return feature_actions


def create_feature_handler(
    df: pd.DataFrame, outdir: str, feature_actions_file: str
) -> FeatureHandler:
    feature_actions = load_feature_actions(feature_actions_file)
    feature_handler = FeatureHandler.from_df(df, feature_actions)
    create_output_dir(outdir)
    feature_handler.set_user_actions()
    feature_handler.to_json(outdir)
    feature_handler.pretty_print_to_json(outdir)
    return feature_handler


def select_features(df: pd.DataFrame, feature_handler: FeatureHandler) -> None:
    features_autoembedding_numerical = feature_handler.get_selected_features_names(
        include_features_with_flags=["is_numerical"],
        exclude_features_with_flags=["is_constant", "is_index"],
        exclude_features_with_actions=["drop"],
        available_feature_names=df.columns,
    )
    features_autoembedding_categorical = feature_handler.get_selected_features_names(
        include_features_with_flags=["is_non_numerical"],
        exclude_features_with_flags=["is_constant", "is_index"],
        exclude_features_with_actions=["drop", "target"],
        available_feature_names=df.columns,
    )
    other_features = feature_handler.get_selected_features_names(
        include_features_with_actions=["target"],
        available_feature_names=df.columns,
    )
    return (
        features_autoembedding_numerical,
        features_autoembedding_categorical,
        other_features,
    )


def load_features(
    df: pd.DataFrame, feature_handler_file: str
) -> Tuple[list[str], list[str]]:
    feature_handler = FeatureHandler.from_json(feature_handler_file)
    numerical_features, categorical_features, target_features = select_features(
        df, feature_handler
    )
    return numerical_features, categorical_features, target_features


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


def prepare_data(df: pd.DataFrame, params: dict[str]) -> list:

    numerical_features, categorical_features, target_features = load_features(
        df, params["feature_handler_file"]
    )

    df_encoded, encoding_reference_values = encode_categorical_input_ordinal(
        df[categorical_features + target_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(
        df[numerical_features], method=params["normalisation_method"]
    )
    df = pd.concat([df_numericals_normalised, df_encoded], axis=1)

    train_df, test_df = train_test_split(df, test_size=params["test_data_fraction"])

    train_df_num, train_df_cat, train_df_target = (
        train_df[numerical_features],
        train_df[categorical_features],
        tf.keras.utils.to_categorical(train_df[target_features]),
    )
    test_df_num, test_df_cat, test_df_target = (
        test_df[numerical_features],
        test_df[categorical_features],
        tf.keras.utils.to_categorical(test_df[target_features]),
    )

    encoding_reference_values_target = [
        encoding_reference_values.pop(target) for target in target_features
    ]
    return (
        train_df_num,
        train_df_cat,
        test_df_num,
        test_df_cat,
        train_df_target,
        test_df_target,
        encoding_reference_values,
        encoding_reference_values_target,
    )
