"""

"""
import tensorflow as tf


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
        embedding_layer_output,
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
                embedding_layer_output[:, idx],
                tf.convert_to_tensor(matched_category_idx),
            )

    def result(self):
        # result(self), which uses the state variables to compute the final results.
        return self.confusion_metric_dict

    def reset_state(self):
        # reset_state(self), which reinitializes the state of the metric.
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_metric_dict = {}