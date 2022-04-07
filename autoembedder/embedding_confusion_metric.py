"""

"""
import tensorflow as tf
import pdb


class EmbeddingConfusionMetric(tf.keras.metrics.Metric):
    def __init__(self, name="EmbeddingConfusionMetric", **kwargs):
        # __init__(self), in which you will create state variables for your metric.
        super(EmbeddingConfusionMetric, self).__init__(name=name, **kwargs)
        self._accuracy = tf.keras.metrics.Accuracy()
        from sklearn.metrics import precision_score

        self._precision = precision_score
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

        # update_state(self, y_true, y_pred, sample_weight=None), which uses the
        # targets y_true and the model predictions y_pred to update the state
        # variables.
        for idx, embedded_feature_batch in enumerate(embedding_layer_outputs_reco):
            # island
            this_feature_name = list(embeddings_reference_values.keys())[idx]

            reference_embeddings = tf.convert_to_tensor(
                list(embeddings_reference_values.values())[idx], dtype=tf.float64
            )
            matched_category_idx = []
            distances = []
            for emb_ref in reference_embeddings:
                distances.append(
                    self._cosine_loss(
                        tf.cast(embedded_feature_batch, tf.float64),
                        tf.cast(emb_ref, tf.float64),
                    )
                )
            matched_category_idx = tf.math.argmin(
                tf.transpose(tf.convert_to_tensor(distances)), axis=1
            )
            self.confusion_metric_dict[
                f"{this_feature_name}_accuracy"
            ] = self._accuracy(
                tf.cast(embedding_layer_input[:, idx], dtype=tf.int64),
                tf.convert_to_tensor(matched_category_idx),
            )

            # self.confusion_metric_dict[
            #     f"{this_feature_name}_precision"
            # ] = self._precision(
            #     tf.cast(embedding_layer_input[:, idx], dtype=tf.int64),
            #     tf.convert_to_tensor(matched_category_idx),
            #     average="micro",
            # )

    def result(self):
        # result(self), which uses the state variables to compute the final results.
        return self.confusion_metric_dict

    def reset_state(self):
        # reset_state(self), which reinitializes the state of the metric.
        # The state of the metric will be reset at the start of each epoch.
        self.confusion_metric_dict = {}