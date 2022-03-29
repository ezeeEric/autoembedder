import os
import glob
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple


from autoembedder.autoembedder import AutoEmbedder
from scripts.train_autoembedder import load_features, prepare_data_for_fit


def create_output_dir(out_dir: str) -> str:
    """Create a given output directory if it does not exist."""
    target_dir = os.path.join(".", out_dir, "")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


def save_model(
    auto_embedder: AutoEmbedder, output_directory: str, model_name: str = "autoembedder"
) -> None:
    create_output_dir(output_directory)
    output_file = os.path.join(output_directory, model_name)
    auto_embedder.save(output_file)


def load_model(model_dir: str) -> AutoEmbedder:
    try:
        return tf.keras.models.load_model(
            model_dir, custom_objects={"AutoEmbedder": AutoEmbedder}
        )
    except OSError:
        print(f"Warning: No model found in {model_dir}")
    return None


def get_sorted_input_files(
    input_dir: str, input_patterns: list[str], input_extension: str = "feather"
) -> list[str]:
    if input_patterns is None or input_patterns == "" or input_patterns == []:
        return sorted(glob.glob(f"{input_dir}/*.{input_extension}"), reverse=True)
    else:
        input_files = []
        for pattern in input_patterns:
            these_input_files = sorted(
                glob.glob(f"{input_dir}/*{pattern}*.{input_extension}")
            )
            input_files.extend(these_input_files)
        return input_files


def get_dtype_dict(df: pd.DataFrame) -> Dict[str, list[str]]:
    """Create a dictionary for numerical and categorical columns.

    For given dataframe, return a dict where the keys 'numerical' and 'categorical'
    contain lists of all the numerical / categorical column names, respectively.
    """
    df_type_dict = {"numerical": [], "categorical": []}

    for col_name in df.columns:
        if pd.api.types.is_numeric_dtype(df[col_name]):
            df_type_dict["numerical"].append(col_name)
        else:
            df_type_dict["categorical"].append(col_name)
    return df_type_dict


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
