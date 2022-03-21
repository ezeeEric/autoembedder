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
from scripts.supervised_classification_wembedding import (
    SimpleClassificationNetwork,
    run_simple_classification,
    evaluate_simple_classification,
    prepare_penguin_data,
)
from autoembedder.embedder import Embedder
from scripts.test_autoembedder import get_model

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
        # feed through embedder

        # get output and feed to network

        input_cat_enc = inputs[0]
        input_num = tf.cast(inputs[1], dtype=tf.float64)
        embedded_input = self._forward_call_embedder(input_cat_enc)
        full_input = tf.concat([embedded_input, input_num], axis=1)
        return self.classification_model(full_input)


@with_params("params.yaml", "train_model")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns="",
        input_extension="feather",
    )

    df = pd.read_feather(input_files[0])

    autoembedder = get_model(sys.argv[2])

    (
        train_df_num,
        train_df_cat,
        test_df_num,
        test_df_cat,
        train_df_target,
        test_df_target,
        num_feat,
        encoding_reference_values,
    ) = prepare_penguin_data(df, params)

    model = SimpleClassificationNetwork(
        n_numerical_inputs=len(num_feat),
        encoding_reference_values=encoding_reference_values,
        config=params,
    )

    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
