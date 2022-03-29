"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")

from utils.params import with_params
from utils.utils import get_sorted_input_files, load_model

from scripts.preprocess_data import prepare_penguin_data
from models.base_classification_network import BaseClassificationNetwork
from models.autoembedder_classification_model import AutoEmbedderClassificationModel


OUTPUT_DIRECTORY = ""

# TODO steer me
def run_simple_classification(
    train_data_num, train_data_cat, target_data, model
) -> tf.keras.Model:
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
        metrics=["accuracy"],
    )
    model.fit([train_data_cat, train_data_num], target_data, epochs=100, verbose=1)
    pass


def evaluate_simple_classification(
    model, test_df_num, test_df_cat, test_target
) -> None:
    loss, accuracy = model.evaluate([test_df_cat, test_df_num], test_target, verbose=0)
    print(f" Model loss on the test set: {loss}")
    print(f" Model accuracy on the test set: {100*accuracy}")


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
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
