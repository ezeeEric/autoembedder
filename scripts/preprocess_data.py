import pandas as pd
import tensorflow as tf

from palmerpenguins import load_penguins

from utils.feature_handler import (
    FeatureHandler,
    create_feature_handler,
    select_features,
    load_features,
)
from utils.params import with_params
from utils.utils import create_output_dir, prepare_data_for_fit


OUTPUT_DIR = "data/training_input"
OUTPUT_NAME = "train_data"


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


def filter_columns(df: pd.DataFrame, feature_handler: FeatureHandler) -> pd.DataFrame:
    numerical_cols, categorical_cols, other_cols = select_features(df, feature_handler)
    return df[numerical_cols + categorical_cols + other_cols]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """For the training of the autoembedder, we need a file containing all
    unique entries wrt the selected columns across the reports."""
    return df.drop_duplicates(ignore_index=True)


def load_data() -> pd.DataFrame:
    return load_penguins()


def save_df(df: pd.DataFrame) -> None:
    create_output_dir(OUTPUT_DIR)
    df.to_feather(f"{OUTPUT_DIR}/{OUTPUT_NAME}.feather")


def drop_entries_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0).reset_index(drop=True)


@with_params("params.yaml", "data_preprocessing")
def main(params: dict):
    df = load_data()

    feature_handler = create_feature_handler(
        df, params["feature_handler_dir"], params["feature_action_file"]
    )
    df = remove_duplicates(df)
    df = filter_columns(df, feature_handler)
    df = drop_entries_with_nan(df)
    save_df(df)


if __name__ == "__main__":
    main()