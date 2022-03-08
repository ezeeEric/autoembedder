import sys
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
from palmerpenguins import load_penguins
from utils.feature_handler import FeatureHandler
from utils.params import with_params
from utils.utils import create_output_dir


OUTPUT_DIR = "training_input"
OUTPUT_NAME = "train_data"


def select_features(df: pd.DataFrame, feature_handler: FeatureHandler) -> None:
    features_autoembedding_categorical = feature_handler.get_selected_features_names(
        include_features_with_flags=["is_numerical"],
        exclude_features_with_flags=["is_constant", "is_index"],
        exclude_features_with_actions=["drop"],
        available_feature_names=df.columns,
    )
    features_autoembedding_numerical = feature_handler.get_selected_features_names(
        include_features_with_flags=["is_non_numerical"],
        exclude_features_with_flags=["is_constant", "is_index"],
        exclude_features_with_actions=["drop"],
        available_feature_names=df.columns,
    )
    return features_autoembedding_numerical, features_autoembedding_categorical


def filter_columns(df: pd.DataFrame, feature_handler: FeatureHandler) -> pd.DataFrame:
    numerical_cols, categorical_cols = select_features(df, feature_handler)
    return df[numerical_cols + categorical_cols]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """For the training of the autoembedder, we need a file containing all
    unique entries wrt the selected columns across the reports."""
    return df.drop_duplicates(ignore_index=True)


def load_data() -> pd.DataFrame:
    return load_penguins()


def create_feature_handler(df: pd.DataFrame, outdir: str) -> FeatureHandler:
    feature_handler = FeatureHandler.from_df(df)
    create_output_dir(outdir)
    feature_handler.to_json(outdir)
    feature_handler.pretty_print_to_json(outdir)
    return feature_handler


def save_df(df: pd.DataFrame) -> None:
    create_output_dir(OUTPUT_DIR)
    df.to_feather(f"{OUTPUT_DIR}/{OUTPUT_NAME}.feather")


def drop_entries_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0).reset_index(drop=True)


@with_params("params.yaml", "data_preprocessing")
def main(params: dict):
    df = load_data()
    feature_handler = create_feature_handler(df, params["feature_handler_dir"])
    df = remove_duplicates(df)
    df = filter_columns(df, feature_handler)
    df = drop_entries_with_nan(df)
    save_df(df)


if __name__ == "__main__":
    main()