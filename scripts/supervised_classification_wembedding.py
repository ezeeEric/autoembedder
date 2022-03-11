"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Tuple
import tensorflow as tf
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

print(f"tensorflow {tf.__version__}")

# from autoembedder.autoembedder import load_features,
from utils.feature_handler import FeatureHandler
from utils.params import with_params
from utils.utils import create_output_dir, get_sorted_input_files
from scripts.preprocess_data import select_features
from scripts.train_autoembedder import load_features, prepare_data_for_fit

OUTPUT_DIRECTORY = ""


def run_simple_classification(X, y) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(len(X.columns), activation="relu"),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
        metrics=["accuracy"],
    )
    model.fit(X, y, epochs=100, verbose=0)
    return model


def evaluate_simple_classification(model, X_test, y_test) -> None:
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f" Model loss on the test set: {loss}")
    print(f" Model accuracy on the test set: {100*accuracy}")


@with_params("params.yaml", "train_model")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[2].split(",") if len(sys.argv) > 2 else None,
        input_extension="feather",
    )
    # create_output_dir(OUTPUT_DIRECTORY)

    df = pd.read_feather(input_files[0])

    numerical_features, categorical_features = load_features(
        df, params["feature_handler_file"]
    )
    train_df, test_df, encoding_reference_values = prepare_data_for_fit(
        df, numerical_features, categorical_features, params
    )

    train_df_target = tf.keras.utils.to_categorical(train_df.pop("species"))
    test_df_target = tf.keras.utils.to_categorical(test_df.pop("species"))
    num_feat = ["bill_depth_mm", "bill_length_mm", "body_mass_g", "flipper_length_mm"]
    train_df = train_df[num_feat]
    test_df = test_df[num_feat]
    model = run_simple_classification(train_df, train_df_target)
    evaluate_simple_classification(model, test_df, test_df_target)


if __name__ == "__main__":
    main()
