import pandas as pd

from palmerpenguins import load_penguins

from utils.feature_handler import FeatureHandler
from utils.data import create_feature_handler, select_features
from utils.params import with_params
from utils.utils import create_output_dir


OUTPUT_DIR = "data/training_input"
OUTPUT_NAME = "train_data"


def filter_columns(df: pd.DataFrame, feature_handler: FeatureHandler) -> pd.DataFrame:
    numerical_cols, categorical_cols, other_cols = select_features(df, feature_handler)
    return df[numerical_cols + categorical_cols + other_cols]


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """For the training of the autoembedder, we need a file containing all
    unique entries wrt the selected columns across the reports."""
    return df.drop_duplicates(ignore_index=True)


def load_data(config: dict) -> pd.DataFrame:

    if config["dataset_tag"] == "penguins":
        return load_penguins()
    elif config["dataset_tag"] == "adults":
        indir = config["dataset_path"]
        return pd.read_csv(indir, sep=",", index_col=False)
    else:
        raise NotImplementedError(f"Unknown dataset {config['dataset_tag']}.")


def save_df(df: pd.DataFrame) -> None:
    create_output_dir(OUTPUT_DIR)
    df.to_feather(f"{OUTPUT_DIR}/{OUTPUT_NAME}.feather")


def drop_entries_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(axis=0).reset_index(drop=True)


@with_params("params.yaml", "data_preprocessing")
def main(params: dict):
    df = load_data(params)
    feature_handler = create_feature_handler(
        df, params["feature_handler_dir"], params["feature_action_file"]
    )
    df = remove_duplicates(df)
    df = filter_columns(df, feature_handler)
    df = drop_entries_with_nan(df)
    save_df(df)


if __name__ == "__main__":
    main()