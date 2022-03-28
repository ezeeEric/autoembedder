import tensorflow as tf
from autoembedder.embedder import Embedder


class BaseClassificationNetwork(Embedder):
    def __init__(self, n_numerical_inputs: int, **kwargs) -> None:
        super().__init__(**kwargs)

        input_length = n_numerical_inputs + sum(self.embedding_layers_output_dimensions)
        n_nodes_per_layer = [input_length] + self.config["n_nodes_per_layer"]

        self.classification_model = tf.keras.Sequential(
            name="base_classification_model"
        )
        for n_nodes in n_nodes_per_layer[:-1]:
            self.classification_model.add(
                tf.keras.layers.Dense(
                    n_nodes, activation=self.config["hidden_layer_activation_function"]
                )
            )

        self.classification_model.add(
            tf.keras.layers.Dense(
                n_nodes_per_layer[-1],
                activation=self.config["output_layer_activation_function"],
            )
        )

    def call(self, inputs: list[tf.Tensor], training: bool = None) -> tf.Tensor:
        """Forward call of this model. Data is passed through the following
        sequence in training and evaluation. As we use this model as parent
        class and overwrite the call() in its children but do need the forward
        call logic as well, the forward call is living in another method."""
        embedded_input = self._forward_call_embedder(inputs[0])
        input_num = tf.cast(inputs[1], dtype=tf.float64)
        full_input = tf.concat([embedded_input, input_num], axis=1)
        return self.classification_model(full_input)
