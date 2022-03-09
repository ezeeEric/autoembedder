import glob
import os

from typing import Dict
import pandas as pd


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


def create_output_dir(out_dir: str) -> str:
    """Create a given output directory if it does not exist."""
    target_dir = os.path.join(".", out_dir, "")
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


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