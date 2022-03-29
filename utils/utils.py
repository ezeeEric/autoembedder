import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, Tuple

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

from autoembedder.autoembedder import AutoEmbedder


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


def prepare_data_for_fit(
    df: pd.DataFrame,
    numerical_features: list[str],
    categorical_features: list[str],
    normalisation_method: str,
    test_data_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, OrdinalEncoder]:
    """This function first encodes the categorical input, then normalises the numerical input and finally merges the result."""
    df_encoded, embedding_encoder = encode_categorical_input_ordinal(
        df[categorical_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(
        df[numerical_features], method=normalisation_method
    )
    df = pd.concat([df_numericals_normalised, df_encoded], axis=1)
    train_df, test_df = train_test_split(df, test_size=test_data_fraction)
    return train_df, test_df, embedding_encoder
