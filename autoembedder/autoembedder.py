"""
Embedding categorial data by training an embedding layer in an unsupervised way
- using an Autoencoder. Keras is used as backend.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

from autoembedder.embedder import Embedder
from sklearn.preprocessing import OrdinalEncoder


def determine_autoencoder_shape(
    emb_out_dim: list[int], numerical_features: list[int]
) -> list[int]:
    # TODO this shape is chosen for easy dev purposes, needs to be optimised and
    # maybe steered via params
    input_layer_size = sum(emb_out_dim) + len(numerical_features)
    encoder_shape = [
        input_layer_size,
        input_layer_size // 2,
        input_layer_size // 3,
        input_layer_size // 4,
    ]
    decoder_shape = [input_layer_size // 3, input_layer_size // 2, input_layer_size]
    return encoder_shape, decoder_shape


def create_dense_layers(shape: list[int], name: str) -> list[layers.Dense]:
    # TODO activation function as parameter
    return [
        layers.Dense(n_nodes, activation="relu", name=f"{name}_{layer_idx}")
        for layer_idx, n_nodes in enumerate(shape)
    ]


def prepare_data_for_embedder(
    input: tf.Tensor, numerical_columns_idx: list[int], encoded_columns_idx: list[int]
) -> list[tf.Tensor]:
    embedding_layer_input = tf.gather(input, indices=encoded_columns_idx, axis=1)
    numerical_input = tf.gather(input, indices=numerical_columns_idx, axis=1)
    return numerical_input, embedding_layer_input


def prepare_data_for_encoder(
    numerical_input: tf.Tensor, embedding_layer_output: tf.Tensor
) -> tf.Tensor:
    return tf.concat([numerical_input, embedding_layer_output], axis=1)


def load_ordinal_encoder_for_feature(
    encoding_dict: dict[list], feature_name: str
) -> OrdinalEncoder:
    encoder = OrdinalEncoder()
    encoder.categories_ = [np.array(encoding_dict[feature_name], dtype=object)]
    # TODO The following is a hack. The loading and saving of the OrdinalEncoder by only
    # using the dictionary above requires this _missing_indices to be set.
    encoder._missing_indices = {}
    return encoder


def find_embedding_layer_idx_for_feature(
    feature_list: list[str], feature_name: str
) -> int:
    return list(feature_list).index(feature_name)


def apply_ordinal_encoding_column(
    df: pd.Series, encoding_dictionary: dict
) -> pd.DataFrame:
    embedding_input_encoder = load_ordinal_encoder_for_feature(
        encoding_dictionary, df.name
    )
    data_enc = embedding_input_encoder.transform(df.values.reshape(-1, 1))
    return pd.Series(data_enc.flatten(), name=df.name)


class AutoEmbedder(Embedder):
    def __init__(
        self,
        encoding_dictionary: dict[str, list],
        numerical_features: list[str] = [],
        categorical_features: list[int] = [],
        feature_idx_map: dict[str, list] = {},
        embedding_layer_idx_feature_name_map: dict[str, int] = {},
    ):
        """Class for unsupervised categorial feature embedding. Consists of an
        embedding layer and a symmetric autoencoder.
        """
        super().__init__(encoding_dictionary=encoding_dictionary)

        self.numerical_features = numerical_features
        self.categorical_features = categorical_features

        self.feature_idx_map = feature_idx_map
        self.embedding_layer_idx_feature_name_map = embedding_layer_idx_feature_name_map
        self.encoder_shape, self.decoder_shape = determine_autoencoder_shape(
            self.embedding_layers_output_dimensions,
            self.numerical_features,
        )

        self.encoder = self.setup_encoder(self.encoder_shape)
        self.decoder = self.setup_decoder(self.decoder_shape)

    def get_config(self):
        config_dict = {
            "numerical_features": self.numerical_features,
            "categorical_features": self.categorical_features,
            "encoding_dictionary": self.encoding_dictionary,
            "feature_idx_map": self.feature_idx_map,
            "embedding_layer_idx_feature_name_map": self.embedding_layer_idx_feature_name_map,
        }
        return config_dict

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def setup_encoder(self, encoder_shape: list[int]) -> tf.keras.Sequential:
        encoder_layers = create_dense_layers(shape=encoder_shape, name="encoder")
        return tf.keras.Sequential(encoder_layers, name="encoder")

    def setup_decoder(self, decoder_shape: list[int]) -> tf.keras.Sequential:
        dec_layers = create_dense_layers(shape=decoder_shape[:-1], name="decoder")

        # the last layer is a sigmoid activation
        dec_layers.append(
            layers.Dense(
                decoder_shape[-1],
                activation="sigmoid",
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

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss}

    def test_step(self, input_data: tf.Tensor) -> dict:
        # TODO this gets called with model.evaluate(). Can we devise a method to
        # judge the quality of our embedding?
        pass

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
            input_column, self.encoding_dictionary
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
