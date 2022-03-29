"""Train the AutoEmbedder Model.

python train_autoembedder.py ./train_input/
"""

import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")

from utils.params import with_params
from utils.utils import get_sorted_input_files, load_model, prepare_penguin_data
from scripts.supervised_classification_wembedding import (
    run_simple_classification,
    evaluate_simple_classification,
)
from models.autoembedder_classification_model import AutoEmbedderClassificationModel


OUTPUT_DIRECTORY = ""


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

    autoembedder = load_model(sys.argv[2])

    model = AutoEmbedderClassificationModel(
        n_numerical_inputs=len(train_df_num.columns),
        autoembedder=autoembedder,
        encoding_reference_values=encoding_reference_values,
        config=params,
    )
    run_simple_classification(train_df_num, train_df_cat, train_df_target, model)
    evaluate_simple_classification(model, test_df_num, test_df_cat, test_df_target)


if __name__ == "__main__":
    main()
