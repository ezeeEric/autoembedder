import sys
import pandas as pd

from mldq.simple_script import SimpleMultiFilesScript
from mldq.feature_handler import FeatureHandler


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


class ConcatScript(SimpleMultiFilesScript):
    def post_init(self):
        print(
            f"Loading feature specifications from {self.params['feature_handler_file']}"
        )
        self.feature_handler = FeatureHandler.from_json(
            self.params["feature_handler_file"]
        )

    def process_dataframes(self, df_list: list, input_files: list) -> pd.DataFrame:
        file_list = "\n".join(input_files)
        print(f"Merging:\n{file_list}")

        filtered_df_list = [filter_columns(df, self.feature_handler) for df in df_list]
        concat_df = pd.concat(filtered_df_list, ignore_index=True, join="inner")
        concat_df = remove_duplicates(concat_df)

        return concat_df


if __name__ == "__main__":
    ConcatScript(
        input_dir=sys.argv[1],
        output_dir="train_input",
        output_name="train_report",
        param_section="prepare_train_input",
    ).run()
