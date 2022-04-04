"""
Embedding categorial data by training an embedding layer in an unsupervised way
- using an Autoencoder. Keras is used as backend.
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from typing import Tuple

from autoembedder.embedder import Embedder
from autoembedder.embedding_confusion_metric import EmbeddingConfusionMetric

from sklearn.preprocessing import OrdinalEncoder


class AutoembedderCallbacks(tf.keras.callbacks.Callback):
    def __init__(self, validation_data: Tuple[tf.Tensor, tf.Tensor]) -> None:
        super().__init__()
        self.validation_data = validation_data

    # https://www.tensorflow.org/guide/keras/custom_callback
    def on_epoch_end(self, epoch, logs=None):
        metrics = self.model.test_step(input_batch=self.validation_data)
        for metric_name, metric_val in metrics.items():
            print(f"{metric_name}: {metric_val.numpy().item():.2f}")

    def on_train_end(self, logs=None):
        # print(f"\ninput_batch\n {self.model.last_input_output[0]}")
        # print(f"\numerical_input\n {self.model.last_input_output[1]}")
        # print(f"\nembedding_layer_input\n {self.model.last_input_output[2]}")
        # print(f"\nembedding_layer_output\n {self.model.last_input_output[3]}")
        # print(f"\nauto_encoder_input\n {self.model.last_input_output[4]}")
        # print(f"\nauto_encoder_output\n {self.model.last_input_output[5]}")
        # print(f"\nnumerical_input_reco\n {self.model.last_input_output[6]}")
        # print(f"\nembedding_layer_outputs_reco\n {self.model.last_input_output[7]}")
        # print(f"\nembeddings_reference_values\n {self.model.last_input_output[8]}")
        pass

    def on_test_begin(self, logs=None):
        self.model._embedding_confusion_metric.reset_state()
        self.model.create_embedding_reference()


def determine_autoencoder_shape(
    embedding_layers_output_dimensions: list[int],
    n_numerical_features: int,
    config: dict,
) -> list[int]:
    input_layer_size = sum(embedding_layers_output_dimensions) + n_numerical_features

    encoder_shape = [
        input_layer_size,
        input_layer_size // 2,
        input_layer_size // 3,
        input_layer_size // 4,
    ]
    if config["manual_encoder_architecture"]:
        encoder_shape = [input_layer_size] + list(config["manual_encoder_architecture"])

    decoder_shape = list(reversed(encoder_shape[:-1]))
    if config["manual_decoder_architecture"]:
        decoder_shape = list(config["manual_decoder_architecture"]) + [input_layer_size]
    return encoder_shape, decoder_shape


def create_dense_layers(
    shape: list[int], name: str, activation_fct: str
) -> list[tf.keras.layers.Dense]:
    return [
        tf.keras.layers.Dense(
            n_nodes, activation=activation_fct.lower(), name=f"{name}_{layer_idx}"
        )
        for layer_idx, n_nodes in enumerate(shape)
    ]


def prepare_data_for_embedder(
    input: tf.Tensor, numerical_columns_idx: list[int], encoded_columns_idx: list[int]
) -> list[tf.Tensor]:
    numerical_input = tf.gather(input, indices=numerical_columns_idx, axis=1)
    embedding_layer_input = tf.gather(input, indices=encoded_columns_idx, axis=1)
    return numerical_input, embedding_layer_input


def prepare_data_for_encoder(
    numerical_input: tf.Tensor, embedding_layer_output: tf.Tensor
) -> tf.Tensor:
    return tf.concat([numerical_input, embedding_layer_output], axis=1)


def split_autoencoder_output(
    input_data: tf.Tensor, n_numerical_nodes: int, embedding_output_split: list[int]
) -> tf.Tensor:
    # this is the reverse operation of prepare_data_for_encoder()
    split_tensors = tf.split(
        input_data, [n_numerical_nodes] + embedding_output_split, axis=1
    )
    return split_tensors[0], split_tensors[1:]


def load_ordinal_encoder_for_feature(
    encoding_dict: dict[list], feature_name: str
) -> OrdinalEncoder:
    encoder = OrdinalEncoder()
    encoder.categories_ = [np.array(encoding_dict[feature_name], dtype=object)]
    # Loading the OrdinalEncoder like this requires the variable below to be set.
    encoder._missing_indices = {}
    return encoder


def find_embedding_layer_idx_for_feature(
    feature_list: list[str], feature_name: str
) -> int:
    return list(feature_list).index(feature_name)


def apply_ordinal_encoding_column(
    df: pd.Series, encoding_reference_values: dict
) -> pd.DataFrame:
    embedding_input_encoder = load_ordinal_encoder_for_feature(
        encoding_reference_values, df.name
    )
    data_enc = embedding_input_encoder.transform(df.values.reshape(-1, 1))
    return pd.Series(data_enc.flatten(), name=df.name)


class AutoEmbedder(Embedder):
    def __init__(
        self,
        numerical_features: list[str] = [],
        categorical_features: list[int] = [],
        feature_idx_map: dict[str, list] = {},
        **kwargs,
    ):
        """Class for unsupervised categorial feature embedding. Consists of an
        embedding layer and a symmetric autoencoder.
        """
        super().__init__(**kwargs)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.feature_idx_map = feature_idx_map

        # TODO should this be a member (issues when loading from disk?)
        # this requires to set the @metric property and overwrite it in all child classes
        self._loss_tracker_epoch = tf.keras.metrics.Mean(name="epoch_loss")
        self._embedding_confusion_metric = EmbeddingConfusionMetric()
        self.embeddings_reference_values = {}

        self.encoder_shape, self.decoder_shape = determine_autoencoder_shape(
            self.embedding_layers_output_dimensions,
            len(self.numerical_features),
            self.config,
        )

        self.encoder = self.setup_encoder(self.encoder_shape)
        self.decoder = self.setup_decoder(self.decoder_shape)

        self.last_input_output = []

    def __str__(self) -> str:
        out_string = "Autoembedder\n"
        out_string += "\n".join([f"{key}:{val}" for key, val in self.__dict__.items()])
        return f"{out_string}\n{super().__str__()}"

    def get_config(self):
        config_dict = {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "encoding_reference_values": self.encoding_reference_values,
            "embeddings_reference_values": self.embeddings_reference_values,
            "feature_idx_map": self.feature_idx_map,
            "config": self.config,
        }
        return config_dict

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def setup_encoder(self, encoder_shape: list[int]) -> tf.keras.Sequential:
        encoder_layers = create_dense_layers(
            shape=encoder_shape,
            name="encoder",
            activation_fct=self.config["hidden_layer_activation_function"],
        )
        return tf.keras.Sequential(encoder_layers, name="encoder")

    def setup_decoder(self, decoder_shape: list[int]) -> tf.keras.Sequential:
        dec_layers = create_dense_layers(
            shape=decoder_shape[:-1],
            name="decoder",
            activation_fct=self.config["hidden_layer_activation_function"],
        )

        # the last layer is a tanh activation
        dec_layers.append(
            tf.keras.layers.Dense(
                decoder_shape[-1],
                activation=self.config["hidden_layer_activation_function"],
                name=f"decoder_{len(decoder_shape)}",
            )
        )
        return tf.keras.Sequential(dec_layers, name="decoder")

    def match_feature_to_input_column_idx(self, columns: list[str]):
        """Gets the matching between column index and the available feature
        names. This is needed to match columns with feature names after type
        conversions to tf.Tensors.
        """
        self.feature_idx_map["numerical"] = [
            list(columns).index(feature_name)
            for feature_name in self.numerical_features
        ]
        self.feature_idx_map["categorical"] = [
            list(columns).index(feature_name)
            for feature_name in self.categorical_features
        ]

    def create_embedding_reference(
        self,
    ) -> None:
        for idx, (feature_name, feature_vocabulary) in enumerate(
            self.encoding_reference_values.items()
        ):
            # here we take advantage that entries in the encoding dictionary are lists with persistent order;
            # the positions (idx) of their entries corresponds to the entry's ordinal encoding
            feat_voc_idx = range(len(feature_vocabulary))
            feat_voc_idx_tensor = tf.convert_to_tensor(feat_voc_idx, name=feature_name)
            # feed through embedding layer
            assert (
                feature_name in self.embedding_layers[idx].name
            ), f"Embedding layer name {self.embedding_layers[idx].name} does not match feature name {feature_name}"
            embedded_vocabulary = self.embedding_layers[idx](feat_voc_idx_tensor)
            # store reference
            self.embeddings_reference_values[feature_name] = list(
                map(tuple, embedded_vocabulary.numpy())
            )
        return self.embeddings_reference_values

    def train_step(self, input_batch: tf.Tensor) -> dict:
        """A single training step on a batch of the data. This gets called within model.fit() and is repeated for every batch in every epoch.

        Args:
            input_batch (tf.Tensor): Input data batch with numerical and categorical columns.

        Returns:
            dict: Dictionary of losses and metrics.
        """
        numerical_input, embedding_layer_input = prepare_data_for_embedder(
            input_batch,
            numerical_columns_idx=self.feature_idx_map["numerical"],
            encoded_columns_idx=self.feature_idx_map["categorical"],
        )
        with tf.GradientTape() as tape:
            embedding_layer_output = self._forward_call_embedder(embedding_layer_input)
            auto_encoder_input = prepare_data_for_encoder(
                numerical_input, embedding_layer_output
            )

            auto_encoder_output = self(auto_encoder_input, training=True)
            loss = self.compiled_loss(
                y_true=auto_encoder_input, y_pred=auto_encoder_output
            )

        # split autoencoder output to get reconstructed embedding values for confusion metric
        numerical_input_reco, embedding_layer_outputs_reco = split_autoencoder_output(
            auto_encoder_output,
            numerical_input.shape[1],
            self.embedding_layers_output_dimensions,
        )
        embeddings_reference_values = self.create_embedding_reference()
        self.last_input_output = [
            input_batch,
            numerical_input,
            embedding_layer_input,
            embedding_layer_output,
            auto_encoder_input,
            auto_encoder_output,
            numerical_input_reco,
            embedding_layer_outputs_reco,
            embeddings_reference_values,
        ]

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Compute our own metrics
        self._loss_tracker_epoch.update_state(loss)
        # self._embedding_confusion_metric.update_state(
        #     embedding_layer_input,
        #     embedding_layer_outputs_reco,
        #     embeddings_reference_values,
        # )
        return {
            "loss": self._loss_tracker_epoch.result(),
            **self._embedding_confusion_metric.result(),
        }

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self._loss_tracker_epoch, self._embedding_confusion_metric]

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
        _, embedding_layer_outputs_reco = split_autoencoder_output(
            auto_encoder_output,
            numerical_input.shape[1],
            self.embedding_layers_output_dimensions,
        )
        self._embedding_confusion_metric.update_state(
            embedding_layer_input,
            embedding_layer_outputs_reco,
            self.embeddings_reference_values,
        )

        return self._embedding_confusion_metric.result()

    def predict(self, input_column: pd.Series) -> pd.DataFrame:
        """Takes a categorical column of an input dataframe and returns a
        dataframe with embedded columns. First the data is encoded, then
        passed through the pre-trained embedding layers.

        Args:
            input_column (pd.Series): Single input column with categorical data.

        Returns:
            pd.DataFrame : A dataframe with embedded columns.
        """
        embedding_layer_idx = find_embedding_layer_idx_for_feature(
            self.categorical_features, input_column.name
        )
        embedding_layer_input = apply_ordinal_encoding_column(
            input_column, self.encoding_reference_values
        )
        embedding_layer_output = self._embed_encoded_column(
            tf.convert_to_tensor(embedding_layer_input), embedding_layer_idx
        )

        df_embedded_columns = pd.DataFrame(
            embedding_layer_output.numpy(),
            columns=[
                f"{input_column.name}_emb_{i}"
                for i in range(embedding_layer_output.shape[1])
            ],
        )
        return df_embedded_columns

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """Forward call of this model. Data is passed through the following
        sequence in training and evaluation"""
        x = self.encoder(inputs)
        return self.decoder(x)
