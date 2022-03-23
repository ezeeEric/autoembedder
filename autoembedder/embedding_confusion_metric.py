"""

"""
import numpy as np
import tensorflow as tf
import pandas as pd

from autoembedder.embedder import Embedder
from sklearn.preprocessing import OrdinalEncoder


class EmbeddingConfusionMetric(tf.keras.metrics.Metric):
    def __init__(self, name="EmbeddingConfusionMetric", **kwargs):
        # __init__(self), in which you will create state variables for your metric.
        super(EmbeddingConfusionMetric, self).__init__(name=name, **kwargs)
        self._accuracy = tf.keras.metrics.Accuracy()
        self._cosine_loss = tf.keras.losses.CosineSimilarity(
            axis=-1, reduction=tf.keras.losses.Reduction.NONE
        )

        self.confusion_metric_dict = {}

    def update_state(
        self,
        embedding_layer_input,
        embedding_layer_outputs_reco,
        embeddings_reference_values,
    ):
        # update_state(self, y_true, y_pred, sample_weight=None), which uses the targets y_true and the model predictions y_pred to update the state variables.

        for idx, embedded_feature_batch in enumerate(embedding_layer_outputs_reco):
            this_feature_name = list(embeddings_reference_values.keys())[idx]
            reference_embeddings = tf.convert_to_tensor(
                list(embeddings_reference_values.values())[idx], dtype=tf.float64
            )
            cos_sims = []
            for ref_emb in reference_embeddings:
                cos_dist = self._cosine_loss(
                    ref_emb, tf.cast(embedded_feature_batch, tf.float64)
                )
                cos_sims.append(1 - cos_dist)
            cosine_similarities = tf.transpose(tf.convert_to_tensor(cos_sims))
            matched_category_idx = tf.math.argmax(cosine_similarities, axis=1)

            self.confusion_metric_dict[
                f"{this_feature_name}_accuracy"
            ] = self._accuracy(
                embedding_layer_input[:, idx],
                tf.convert_to_tensor(matched_category_idx),
            )

    def result(self):
        # result(self), which uses the state variables to compute the final results.
        return self.confusion_metric_dict

    def reset_state(self):
        # reset_state(self), which reinitializes the state of the metric.
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_metric_dict = {}

    def test_step(self, input_batch: tf.Tensor) -> dict:
        numerical_input, embedding_layer_input = prepare_data_for_embedder(
            input_batch,
            numerical_columns_idx=self.feature_idx_map["numerical"],
            encoded_columns_idx=self.feature_idx_map["categorical"],
        )
        embedding_layer_output = self._forward_call_embedder(embedding_layer_input)
        assert (
            sum(self.embedding_layers_output_dimensions)
            == embedding_layer_output.shape[1]
        ), "Embedding output layer dimensionality does not match reference"
        auto_encoder_input = prepare_data_for_encoder(
            numerical_input, embedding_layer_output
        )
        auto_encoder_output = self(auto_encoder_input, training=True)

        # split autoencoder output to get reconstructed embedding values
        numerical_input_reco, embedding_layer_outputs_reco = split_autoencoder_output(
            auto_encoder_output,
            numerical_input.shape[1],
            self.embedding_layers_output_dimensions,
        )
        cosine_loss = tf.keras.losses.CosineSimilarity(
            axis=-1, reduction=tf.keras.losses.Reduction.NONE
        )
        # TODO can these loops be replace with tensor operations?
        # matched_category_idx = []
        accuracy = tf.keras.metrics.Accuracy()
        metric_dict = {}
        for idx, embedded_feature_batch in enumerate(embedding_layer_outputs_reco):
            this_feature_name = list(self.embeddings_reference_values.keys())[idx]
            reference_embeddings = tf.convert_to_tensor(
                list(self.embeddings_reference_values.values())[idx], dtype=tf.float64
            )
            cos_sims = []
            for ref_emb in reference_embeddings:
                cos_dist = cosine_loss(
                    ref_emb, tf.cast(embedded_feature_batch, tf.float64)
                )
                cos_sims.append(1 - cos_dist)
            cosine_similarities = tf.transpose(tf.convert_to_tensor(cos_sims))
            matched_category_idx = tf.math.argmax(cosine_similarities, axis=1)
            metric_dict[f"{this_feature_name}_accuracy"] = accuracy(
                embedding_layer_input[:, idx],
                tf.convert_to_tensor(matched_category_idx),
            )

        return metric_dict
