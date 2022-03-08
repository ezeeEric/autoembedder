"""
Test a trained auto-embedder model.
"""
import sys
import pandas as pd
import tensorflow as tf

print(f"tensorflow {tf.__version__}")
from mldq import with_params
from mldq.report_reading import create_output_dir, get_sorted_input_files


from auto_embedding.embedding.auto_embedder import AutoEmbedder

OUTPUT_DIRECTORY = "model_tests"


def get_model(model_dir: str) -> AutoEmbedder:
    try:
        return tf.keras.models.load_model(
            model_dir, custom_objects={"AutoEmbedder": AutoEmbedder}
        )
    except OSError:
        print(f"Warning: No model found in {model_dir}")
    return None


def check_loaded_model(df: pd.DataFrame, model: AutoEmbedder) -> None:
    # TODO implement full test procedure
    # these are print-outs to check loaded model
    print(model)
    print(model.numerical_features)
    print(model.categorical_features)
    print(model.n_categories_per_feature)
    print(model.embedding_layers)
    print(model.embedding_layers_output_dimensions)
    return model


@with_params("params.yaml", "train_model")
def main(params: dict):
    
    input_files = get_sorted_input_files(
        input_dir=sys.argv[1],
        input_patterns=sys.argv[3].split(",") if len(sys.argv) > 3 else "",
        input_extension="feather",
    )
    target_dir = create_output_dir(OUTPUT_DIRECTORY)

    if len(input_files) > 1:
        print(
            "Warning: Multiple test input files not supported. Only first one will be processed."
        )
    df = pd.read_feather(input_files[0])

    model = get_model(model_dir=sys.argv[2])
    check_loaded_model(df, model)


if __name__ == "__main__":
    main()