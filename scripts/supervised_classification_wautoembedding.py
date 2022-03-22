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
from autoembedder.autoembedder import AutoEmbedder
from scripts.test_autoembedder import get_model

OUTPUT_DIRECTORY = ""


class SimpleClassificationNetworkWithEmbedder(SimpleClassificationNetwork):
    def __init__(
        self, n_numerical_inputs: int, autoembedder: AutoEmbedder, **kwargs
    ) -> None:
        super().__init__(n_numerical_inputs, **kwargs)

        self.autoembedder = autoembedder
        input_length = n_numerical_inputs + sum(self.embedding_layers_output_dimensions)

        self.classification_model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(input_length, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )

    def train_step(self, input_batch: tf.Tensor) -> dict:
        (data_cat, data_num), target = input_batch
        # for layer_idx, layer in enumerate(self.embedding_layers):
        #     encoded_column = get_encoded_column(inputs, layer_idx)
        #     embedded_column = layer(encoded_column)
        embedding_layer_output = self.autoembedder._forward_call_embedder(data_cat)
        # print(embedding_layer_output)
        exit()
        # with tf.GradientTape() as tape:
        #     auto_encoder_input = prepare_data_for_encoder(
        #         numerical_input, embedding_layer_output
        #     )

        #     auto_encoder_output = self(auto_encoder_input, training=True)
        #     loss = self.compiled_loss(
        #         y_true=auto_encoder_input, y_pred=auto_encoder_output
        #     )

        # # Compute gradients
        # trainable_vars = self.trainable_variables
        # gradients = tape.gradient(loss, trainable_vars)

        # # Update weights
        # self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # return {"loss": loss}

    def call(self, inputs: list[tf.Tensor], training: bool = None) -> tf.Tensor:
        # feed through embedder

        # get output and feed to network

        input_cat_enc = inputs[0]
        input_num = tf.cast(inputs[1], dtype=tf.float64)
        embedded_input = self._forward_call_embedder(input_cat_enc)
        full_input = tf.concat([embedded_input, input_num], axis=1)
        return self.classification_model(full_input)


@with_params("params.yaml", "train_simple_classification_model")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns="",
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
        num_feat,
        encoding_reference_values,
        encoding_reference_values_target,
    ) = prepare_penguin_data(df, params)

    autoembedder = get_model(sys.argv[2])
    model = SimpleClassificationNetworkWithEmbedder(
        n_numerical_inputs=len(num_feat),
        autoembedder=autoembedder,
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
