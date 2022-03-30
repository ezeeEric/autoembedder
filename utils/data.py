import json
import pandas as pd
from typing import Tuple

import tensorflow as tf

from utils.utils import create_output_dir, prepare_data_for_fit
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


def prepare_penguin_data(
    df: pd.DataFrame,
    params: dict[str],
) -> list:

    numerical_features, categorical_features, target_features = load_features(
        df, params["feature_handler_file"]
    )
    train_df, test_df, encoding_reference_values = prepare_data_for_fit(
        df,
        numerical_features,
        categorical_features + target_features,
        normalisation_method=params["normalisation_method"],
        test_data_fraction=params["test_data_fraction"],
    )

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

    encoding_reference_values_target = encoding_reference_values.pop("species")
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