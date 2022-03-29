"""
If a model is provided using a second argument, the classification is run using the pretrained autoembedder.

- python scripts/supervised_classification_wembedding.py ./data/training_input/ [./data/model/autoembedder]
"""

import sys
import pandas as pd
import tensorflow as tf

from utils.params import with_params
from utils.utils import get_sorted_input_files, load_model

from scripts.preprocess_data import prepare_penguin_data
from models.base_classification_network import BaseClassificationNetwork
from models.autoembedder_classification_model import AutoEmbedderClassificationModel


OUTPUT_DIRECTORY = ""


def compile_model(
    model: tf.keras.Model,
    config: dict,
) -> None:
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=config["learning_rate"]),
        metrics=list(config["metrics"]),
    )


def train_model(
    train_data_num: pd.DataFrame,
    train_data_cat: pd.DataFrame,
    train_data_target: pd.DataFrame,
    model: tf.keras.Model,
    config: dict,
) -> None:

    model.fit(
        [train_data_cat, train_data_num],
        train_data_target,
        epochs=config["n_epochs"],
        verbose=config["verbosity_level"],
    )


def test_model(
    test_data_num: pd.DataFrame,
    test_data_cat: pd.DataFrame,
    test_data_target: pd.DataFrame,
    model: tf.keras.Model,
    config: dict,
) -> None:
    loss, accuracy = model.evaluate(
        [test_data_cat, test_data_num],
        test_data_target,
        verbose=config["verbosity_level"],
    )
    print(f" Model loss on the test set: {loss}")
    print(f" Model accuracy on the test set: {100*accuracy}%")


@with_params("params.yaml", "train_classification_models")
def main(params: dict):

    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns="",
        input_extension="feather",
    )

    df = pd.read_feather(input_files[0])

    (
        train_df_num,
        train_df_cat,
        test_df_num,
        test_df_cat,
        train_df_target,
        test_df_target,
        encoding_reference_values,
        _,
    ) = prepare_penguin_data(df, params)

    if len(sys.argv) > 2:
        print(f"Pretrained Autoembedder {sys.argv[2]} will be used in classification.")
        autoembedder = load_model(sys.argv[2])
        model = AutoEmbedderClassificationModel(
            n_numerical_inputs=len(train_df_num.columns),
            autoembedder=autoembedder,
            encoding_reference_values=encoding_reference_values,
            config=params,
        )
    else:
        print(f"No pretrained model defined, using basic model.")
        model = BaseClassificationNetwork(
            n_numerical_inputs=len(train_df_num.columns),
            encoding_reference_values=encoding_reference_values,
            config=params,
        )
    compile_model(model=model, config=params)
    train_model(
        train_data_num=train_df_num,
        train_data_cat=train_df_cat,
        train_data_target=train_df_target,
        model=model,
        config=params,
    )
    test_model(
        test_data_num=test_df_num,
        test_data_cat=test_df_cat,
        test_data_target=test_df_target,
        model=model,
        config=params,
    )


if __name__ == "__main__":
    main()
