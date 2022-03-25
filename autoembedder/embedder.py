import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
import pandas as pd


def get_encoded_column(input: np.ndarray, slice_idx: int) -> list[np.ndarray]:
    return input[:, slice_idx]


class Embedder(Model):
    def __init__(
        self,
        encoding_reference_values: dict[str, list],
        config: dict,
        embeddings_reference_values: dict[str, list] = {},
    ):
        super().__init__()
        self.config = config

        self.encoding_reference_values = encoding_reference_values
        self.embeddings_reference_values = embeddings_reference_values

        self.n_categories_per_feature = [
            len(encoded_cats) for encoded_cats in encoding_reference_values.values()
        ]

        self.embedding_layers = []
        self.embedding_layers_output_dimensions = []
        for idx, in_dim in enumerate(self.n_categories_per_feature):
            feature_name = list(self.encoding_reference_values)[idx]
            layer, out_dim = self.create_layer(in_dim, feature_name)
            self.embedding_layers.append(layer)
            self.embedding_layers_output_dimensions.append(out_dim)

    def __str__(self) -> str:
        output = [
            f"Embedding layer {layer_idx}: Shape "
            + f"({self.n_categories_per_feature[layer_idx]},"
            + f"{self.embedding_layers_output_dimensions[layer_idx]}) (In,Out)"
            + f"{self.embedding_layers[layer_idx].weights}) (weights)"
            for layer_idx in range(len(self.embedding_layers))
        ]
        output.append(super(Embedder, self).__str__())
        return "\n".join(output)

    def create_layer(
        self, n_categories: int, feature_name: str
    ) -> tf.keras.layers.Embedding:

        # input_size - each embedding layer should have vocabulary size + 1
        # https://github.com/keras-team/keras/issues/3110#issuecomment-345153450
        input_dim = n_categories + 1
        # input_dim = n_categories
        #  + 1

        # output_dim=int(input_size ** 0.25)
        # https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html
        # embedding_output_dimension = int(input_dim ** 0.25)
        embedding_output_dimension = n_categories

        # this assumes we use one layer per input feature. Sentence embeddings
        # would use multiple inputs here; however there's no intrinsic
        # positional relation between our words.
        input_length = 1

        # the initialisation of the embedding layer weights is important for performance
        if self.config["embeddings_initializer"] == "uniform":
            embeddings_initializer = (
                tf.keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None),
            )
        else:
            raise NotImplementedError(
                f"embeddings_initializer {self.config['embeddings_initializer']} not implemented."
            )

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=input_dim,
            output_dim=embedding_output_dimension,
            input_length=input_length,
            dtype=np.float64,
            embeddings_initializer=embeddings_initializer,
            name=f"embedding_{feature_name}",
        )
        return embedding_layer, embedding_output_dimension

    def _embed_encoded_column(
        self, input_data: tf.Tensor, layer_idx: int
    ) -> pd.DataFrame:
        return self.embedding_layers[layer_idx](input_data)

    def _forward_call_embedder(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """Forward call of this model. Data is passed through the following
        sequence in training and evaluation"""
        embedded_inputs = []
        for layer_idx, layer in enumerate(self.embedding_layers):
            encoded_column = get_encoded_column(inputs, layer_idx)
            embedded_column = layer(encoded_column)
            embedded_inputs.append(embedded_column)
        return tf.concat(embedded_inputs, axis=1)

    def call(self, inputs: list[pd.DataFrame], training: bool = None) -> tf.Tensor:
        """Forward call of this model. Data is passed through the following
        sequence in training and evaluation. As we use this model as parent
        class and overwrite the call() in its children but do need the forward
        call logic as well, the forward call is living in another method."""
        return self._forward_call_embedder(inputs)
