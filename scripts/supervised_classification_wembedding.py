"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")

from utils.feature_handler import FeatureHandler
from utils.params import with_params
from utils.utils import get_sorted_input_files
from scripts.train_autoembedder import load_features, prepare_data_for_fit
from models.base_classification_network import BaseClassificationNetwork


OUTPUT_DIRECTORY = ""

# TODO this function should rather be part of main
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


# TODO steer me
def run_simple_classification(
    train_data_num, train_data_cat, target_data, model
) -> tf.keras.Model:
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
        metrics=["accuracy"],
    )
    model.fit([train_data_cat, train_data_num], target_data, epochs=100, verbose=1)
    pass


def evaluate_simple_classification(
    model, test_df_num, test_df_cat, test_target
) -> None:
    loss, accuracy = model.evaluate([test_df_cat, test_df_num], test_target, verbose=0)
    print(f" Model loss on the test set: {loss}")
    print(f" Model accuracy on the test set: {100*accuracy}")


@with_params("params.yaml", "train_simple_classification_model")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[2].split(",") if len(sys.argv) > 2 else None,
        input_extension="feather",
    )

    df = pd.read_feather(input_files[0])

    (
        train_df_num,
        train_df_cat,
        test_df_num,
        test_df_cat,
        train_df_target,
        test_df_target,
        encoding_reference_values,
        _,
    ) = prepare_penguin_data(df, params)

    model = BaseClassificationNetwork(
        n_numerical_inputs=len(train_df_num.columns),
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
