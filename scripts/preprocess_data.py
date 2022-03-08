import sys
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
from palmerpenguins import load_penguins
from feature_handling.feature_handler import FeatureHandler

OUTPUT_DIR = "training_input"
OUTPUT_NAME = "train_data"


def select_features(df: pd.DataFrame, feature_handler: FeatureHandler) -> None:
    features_autoembedding_categorical = feature_handler.get_selected_features_names(
        include_features_with_actions=["auto_embedding_categorical"],
        exclude_features_with_flags=["is_constant", "is_index"],
        available_feature_names=df.columns,
    )
    features_autoembedding_numerical = feature_handler.get_selected_features_names(
        include_features_with_flags=["uses_reporting_currency"],
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


def create_feature_handler(df: pd.DataFrame) -> None:
    feature_handler = FeatureHandler.from_df(df)
    feature_handler.to_json(OUTPUT_DIR)
    feature_handler.pretty_print_to_json(OUTPUT_DIR)


def save_df(df: pd.DataFrame) -> None:
    df.to_feather(f"{OUTPUT_DIR}/{OUTPUT_NAME}.feather")


def main():
    df = load_data()
    create_feature_handler(df)
    save_df(df)


if __name__ == "__main__":
    main()