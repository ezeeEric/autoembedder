import tensorflow as tf
from autoembedder.autoembedder import AutoEmbedder

from models.base_classification_network import BaseClassificationNetwork


class AutoEmbedderClassificationModel(BaseClassificationNetwork):
    def __init__(self, autoembedder: AutoEmbedder, **kwargs) -> None:
        super().__init__(**kwargs)

        self.autoembedder = autoembedder

    @property
    def metrics(self):
        # TODO
        """This resets the metrics property to the tensorflow default. The
        parent class has a special setting only applicable for the
        Autoembedder training.
        """
        return self.compiled_loss.metrics + self.compiled_metrics.metrics

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
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(target, network_output)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

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
