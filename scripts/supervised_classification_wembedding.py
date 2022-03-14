"""Supervised training of a simple classification model using embedding layers.
"""

import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")

from utils.params import with_params
from utils.utils import get_sorted_input_files
from scripts.train_autoembedder import (
    load_features,
    compile_model,
    encode_categorical_input_ordinal,
    normalise_numerical_input_columns,
)
from autoembedder.embedder import Embedder
from sklearn.model_selection import KFold


class SimpleClassificationNetwork(Embedder):
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


def prepare_penguins_for_fit(
    df: tf.Tensor,
    numerical_features: list[str],
    categorical_features: list[str],
) -> tf.Tensor:
    """This function first encodes the categorical input, then normalises the numerical input and finally merges the result."""
    # TODO this is somewhat duplicated with apply_ordinal_encoding_column()
    df_encoded, embedding_encoder = encode_categorical_input_ordinal(
        df[categorical_features]
    )
    df_numericals_normalised = normalise_numerical_input_columns(df[numerical_features])
    return pd.concat([df_numericals_normalised, df_encoded], axis=1), embedding_encoder

    # train_df, test_df = train_test_split(df, test_size=config["test_data_fraction"])
    # return train_df, test_df, embedding_encoder


def create_k_fold_split(X_data, y_data, n_splits):
    def gen():
        for train_index, test_index in KFold(n_splits).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train, y_train, X_test, y_test

    return tf.data.Dataset.from_generator(
        gen, (tf.float64, tf.float64, tf.float64, tf.float64)
    )


# dataset=make_dataset(X,y,10)


def fit_simple_classification_model(
    model,
    train_data_num,
    train_data_cat,
    y,
) -> tf.keras.Model:
    model.fit([train_data_cat, train_data_num], y, epochs=1000, verbose=0)


def evaluate_simple_classification(
    model, test_df_num, test_df_cat, test_target
) -> None:
    loss, accuracy = model.evaluate([test_df_cat, test_df_num], test_target, verbose=0)
    print(f" Model loss on the test set: {loss}")
    print(f" Model accuracy on the test set: {100*accuracy}")


@with_params("params.yaml", "train_simple_classification_model")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[2].split(",") if len(sys.argv) > 2 else None,
        input_extension="feather",
    )

    df = pd.read_feather(input_files[0])

    numerical_features, categorical_features = load_features(
        df, params["feature_handler_file"]
    )
    df, encoding_reference_values = prepare_penguins_for_fit(
        df, numerical_features, categorical_features, params
    )

    train_df_target = tf.keras.utils.to_categorical(train_df.pop("species"))
    test_df_target = tf.keras.utils.to_categorical(test_df.pop("species"))

    num_feat = ["bill_depth_mm", "bill_length_mm", "body_mass_g", "flipper_length_mm"]
    categorical_features.remove("species")
    encoding_reference_values.pop("species")

    model = SimpleClassificationNetwork(
        n_numerical_inputs=len(num_feat),
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    compile_model(
        model=model,
        learning_rate=params["learning_rate"],
        optimizer_name=params["optimizer"],
        loss_name=params["loss"],
        metrics=params["metrics"],
    )
    train_df_num, train_df_cat = train_df[num_feat], train_df[categorical_features]
    test_df_num, test_df_cat = test_df[num_feat], test_df[categorical_features]
    fit_simple_classification_model(model, train_df_num, train_df_cat, train_df_target)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
