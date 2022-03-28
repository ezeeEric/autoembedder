import tensorflow as tf
from autoembedder.embedder import Embedder


class BaseClassificationNetwork(Embedder):
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
