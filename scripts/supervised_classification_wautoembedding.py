"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")

from utils.feature_handler import FeatureHandler
from utils.params import with_params
from utils.utils import get_sorted_input_files, load_model
from scripts.supervised_classification_wembedding import (
    SimpleClassificationNetwork,
    run_simple_classification,
    evaluate_simple_classification,
    prepare_penguin_data,
)
from autoembedder.autoembedder import AutoEmbedder

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
        embedding_layer_output = self.autoembedder._forward_call_embedder(data_cat)
        network_input = tf.concat([data_num, embedding_layer_output], axis=1)
        with tf.GradientTape() as tape:
            network_output = self(network_input, training=True)
            loss = self.compiled_loss(y_true=target, y_pred=network_output)
        # Compute gradients
        trainable_vars = self.trainable_variables
        # https://www.tensorflow.org/api_docs/python/tf/UnconnectedGradients
        gradients = tape.gradient(
            loss, trainable_vars, unconnected_gradients=tf.UnconnectedGradients.ZERO
        )

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

    def test_step(self, input_batch: tf.Tensor) -> dict:
        (data_cat, data_num), target = input_batch
        embedding_layer_output = self.autoembedder._forward_call_embedder(data_cat)
        network_input = tf.concat([data_num, embedding_layer_output], axis=1)
        network_output = self(network_input, training=True)
        # Updates the metrics tracking the loss
        self.compiled_loss(target, network_output, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(target, network_output)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    def call(self, input_data: tf.Tensor, training: bool = None) -> tf.Tensor:
        return self.classification_model(input_data)


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
        encoding_reference_values,
        encoding_reference_values_target,
    ) = prepare_penguin_data(df, params)

    autoembedder = load_model(sys.argv[2])
    model = SimpleClassificationNetworkWithEmbedder(
        n_numerical_inputs=len(train_df_num.columns),
        autoembedder=autoembedder,
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
