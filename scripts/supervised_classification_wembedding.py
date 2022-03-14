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

from utils.feature_handler import FeatureHandler
from utils.params import with_params
from utils.utils import create_output_dir, get_sorted_input_files
from scripts.preprocess_data import select_features
from scripts.train_autoembedder import load_features, prepare_data_for_fit
from scripts.test_autoembedder import get_model

from autoembedder.embedder import Embedder

OUTPUT_DIRECTORY = ""


class SimpleClassificationNetwork(Embedder):
    def __init__(self, n_numerical_inputs: int, **kwargs) -> None:
        super().__init__(**kwargs)

        input_length = n_numerical_inputs + sum(self.embedding_layers_output_dimensions)

        self.classification_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(input_length, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

    def call(self, inputs: list[tf.Tensor], training: bool = None) -> tf.Tensor:
        """Forward call of this model. Data is passed through the following
        sequence in training and evaluation. As we use this model as parent
        class and overwrite the call() in its children but do need the forward
        call logic as well, the forward call is living in another method."""
        input_cat_enc = inputs[0]
        input_num = tf.cast(inputs[1], dtype=tf.float64)
        embedded_input = self._forward_call_embedder(input_cat_enc)
        full_input = tf.concat([embedded_input, input_num], axis=1)
        return self.classification_model(full_input)


def run_simple_classification(
    train_data_num, train_data_cat, y, model
) -> tf.keras.Model:
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
        metrics=["accuracy"],
    )
    model.fit([train_data_cat, train_data_num], y, epochs=100, verbose=1)
    pass


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
    categorical_features.remove("species")
    encoding_reference_values.pop("species")
    model = SimpleClassificationNetwork(
        n_numerical_inputs=len(num_feat),
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    train_df_num, train_df_cat = train_df[num_feat], train_df[categorical_features]
    # test_df = test_df[num_feat + categorical_features]
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    # evaluate_simple_classification(model, test_df, test_df_target)


if __name__ == "__main__":
    main()