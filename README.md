Notes:
```
dvc init
dvc remote add -d gs_dvc_autoembedder gs://mldq/dvc_autoembedder
```

```
import pandas as pd
import seaborn as sns 
from palmerpenguins import load_penguins
sns.set_style('whitegrid')
penguins = load_penguins()
penguins.head()
```

## Autoembedding Categorical Data

In the following, the usage of the Autoembedding model to embed categorical data is described.

:warning: **Disclaimer**: The embedding layers can be trained out of the box with minimal adjustments to the code. However, in order to achieve optimal performance, hyperparameters need to be adjusted for a given dataset/problem. This requires empirical testing.

### Overview

Real-world datasets can include categorical data, which require pre-processing, eg. transformation into numerical values, before being fed to a machine learning algorithm.  While simple approaches like ordinal encoding

> [giraffe, opossum, chipmunk]  -> [1, 2, 3])

or one-hot encoding

> [giraffe, opossum, chipmunk]  -> [ [ 1, 0, 0 ],[ 0, 1, 0 ],[ 0, 0, 1 ]

are readily available, they introduce certain issues: an artificial numerical ordering in the former case and a dimensionality problem in the latter.
Embedding layers can be used as part of Neural Network trainings to mitigate these problems by mapping encoded values to a continuous numerical vector space.

> [giraffe, opossum, chipmunk]  -> [[ 0.12, 0.22,...],[ 0.87, 0.78, ... ],[ 0.11, 0.094, ... ]]

This mapping (eg. weights & biases of embedding layers) are usually learned together with the rest of the model in a supervised fashion. This has the advantage that the embedding is learned as part of the whole model on the complete feature set. The resulting embedding hence reflects the optimal way of translating data for a given oiptimisation target.


The [Autoembedder Model](https://medium.com/kirey-group/autoembedder-training-embedding-layers-on-unsupervised-tasks-fc364c0f6eec) provides a way of training an embedding layer in an **unsupervised manner** by using an Autoencoder as backend.

### The Model

![model sketch](./autoembedder_model_sketch.png?raw=True "The AutoEmbedding Model")

The data is split into categorical (e.g. non-numerical) and  continuous (eg.
numerical, non-discrete) feature sets. The categorical features are encoded
using an alphabetically ordered, ordinal encoding. These encoded features form
the input to the embedding layers. The outputs of these layers are concatenated
with the continuous features and used as input to the AutoEncoder.

This encoder is trained by calculating the loss between its input and output;
the Autoencoder learns how to reconstruct its input. Together with the
parameters of the Autoencoder layers, the embedding layer parameters are
adjusted using backpropagation. This allows for the unsupervised training of
these layers.

Once the training is completed, the Autoencoder is discarded and the embedding
layers can be used to embedd categorical features in unsupervised machine
learning tasks.

### Implementation
The model is implemented using `tensorflow 2.8.0` and `sklearn 1.0`. The files can be found in the directory `./autoembedder/`:

| File  | Description |
| ------------- | ------------- |
| `embedding/embedder.py`  | Class defining and wrapping all Embedding layers  |
| `embedding/auto_embedder.py`  | The AutoEmbedding model; defines Autoencoder architecture and methods used in training and testing |
| `concat_reports.py`  | Script for training input preparation  |
| `train_autoembedder.py`  | Script to steer building, training and saving the model  |
| `test_auto_embedder.py`  | Testing the trained model, loading it from disk |
| `dvc.yaml`  | Definition of pipeline stages  |
| `params.yaml`  | Parameters used in scripts  |

### Usage
The autoembedding `dvc` pipeline defined in `autoembedding/dvc.yaml` includes all relevant steps to use the Autoembedding model:

1. `prepare_train_input`: First we concatenate all reports whilst removing any duplicated rows in order to create a training input file `train_input/train_report.feather`. This file includes all columns of the original reports. Each column has all unique entries available across the reports included. This means the model is trained on the **full vocabulary** of available categories; every possible category is seen once.

2. `train`: This is the main stage of this pipeline, where the model is instantiated, compiled and trained. Before the data can be used for training, it is prepared by selecting a subset of features to use and splitting into continuous and categorical data.  
The categorical data is then ordinal encoded using `sklearn::OrdinalEncoder`, the resulting mapping is a member variable of the Autoembedder instance. The continuous columns are normalized in `normalise_numerical_input_columns` using a manual min-max scaling.   
The model is compiled using an optimizer based on stochastic gradient descent and a Mean Squared Error loss by default, this can be steered via parameters. The modularity of the code allows to add additional optimisers or different computations of the loss function, if required.    
Thereafter, the model is fitted. Since we are converting the dataframe to a `tf.Tensor`, we need to store the mapping of column indices and feature names for later identification. The model can be evaluated after this fit but this feature is currently not implemented. Finally, the model is saved to disk.

3. `test`: Here the model is loaded from disk and the success of this operation is tested by printing the layer configurations. Further testing criteria could be added in the future.


#### Model Parameters

Some model and training configurations are parameters in `autoembedder/params.yaml` and can be steered without changing the codebase. Other parameters are currently hardcoded.

- in `params.yaml`:

| Section |  Parameter  | Description |
| ------------- | ------------- | ------------- |
| `prepare_train_input`  | `feature_handler_file` | This is the summary file of all features processed with the `FeatureHandler` in the `process_features` stage. :warning: Please make sure to run this stage before training the `Autoembedder`, so all feature flags are set properly.  To do this, execute `dvc repro process_features` in the main folder.|
| `train_model`  | `feature_handler_file` | See above  |
|   | `optimizer` | Optimizer to choose during training procedure. Default to stochastic gradient descent.  |
|   | `loss` | Loss function for training. Defaults to Mean Squared Error.  |
|   | `batch_size` | Size of batches.  |
|   | `epochs` | Number of epochs the model is trained for. |
|   | `learning_rate` | Stepsize of updating the model parameters.  |

- in `embedding/embedder.py`: In `create_layer()` several parameters are chosen for the embedding layers, the choice is motivated by sources listed as comments. This includes:

   - the vocabulary size (eg. `input_dim`) which is the number of categories per feature
   - the `embedding_output_dimension`, which corresponds to the number of nodes a given feature is mapped to.
   - the `input_length` which is set to `1` as we treat every feature as individual word and don't consider sequences of words here

- in `embedding/autoembedder.py`: In `determine_autoencoder_shape` the shape of the Autoencoder is automatically determined from the input dimensions of embedding layer output and continuous features combined. A classic, symmetric Autoencoder architecture is chosen using an encoder with 2 hidden layers, a latent layer bottleneck and the inverse of the encoder as decoder. The last layer of the decoder has a `Sigmoid` activation, the other layers use `ReLU`.


### Autoembedding Cookbook: A step-by-step guide

Here follows a simple example on how to use the `Autoembedder` model to embed selected features and use them in the anomaly detection pipeline. 

#### 1. Register Features to use for AutoEmbedding

The categorical features to embed are selected in  `feature_handling/feature_actions.json` in the `autoembedder_categorical` field:
```
"autoembedder_categorical": [
   "cart level 4 code",
   "aim rating",
   "attribute type",
   "coupon"
],
```
The selection of features to use for the categorical and continuous input is then performed in `concat_reports.select_features()`.   
By default, the categorical features registered in `autoembedder_categorical` as described above are chosen, whereas all numerical features with the flag `uses_reporting_currency` are used as continuous features. It is possible to modify this behaviour by manually editing the definitions of `features_autoembedding_categorical` or `features_autoembedding_numerical` in `concat_reports.select_features()`.

The feature actions need to be registered with the `FeatureHandler`, which requires running `dvc repro process_features` in the main directory.


#### 2. Run the Autoembedder training

Having defined the selected features, simply navigate to the subdirectory and run the pipeline
```
cd autoembedder/
dvc repro
```
Adjust the training and model parameters to your liking in `autoembedder/params.yaml`. 

#### 3. Autoembed selected columns

Navigate to the main directory and set the corresponding setting in the `params.yaml` file:

```
use_pretrained_auto_embedder: True 
```

This will load the model from the path set in the field `auto_embedder_model` and embed the previously selected categorical columns. Note that one input feature will yield several embedded columns; the number depending on how many categories are present in that feature.  
The embedded columns follow the naming scheme `original_column_name_emb_0/1/2/...`.

#### 4. Select autoembedded columns for further processing

For selecting autoembedded features in subsequent stages, a summary of column names can be found in `feature_handling/feature_handler_pretty_summary.json`:

```
    "auto_embedded_features": [
        "aim rating_emb_0",
        "aim rating_emb_1",
        "attribute type_emb_0",
        "cart level 4 code_emb_0",
        "cart level 4 code_emb_1",
        "coupon_emb_0"
    ]
```

Select these features or subsets of them for further analysis by adding them in `feature_handling/feature_actions.json` under the field `user_selected`:
```
"user_selected": [
   "cart level 4 code_emb_0",
   "cart level 4 code_emb_1",
   "book yield",
   "book value (rep)",
   "currency_le",
   "mod duration",
   "country of risk_le"
],
```
Rerun the feature processing stage in the main directory to register these variables via 
```
dvc repro process_features
```
You can then launch any or all anomaly detection experiments using the embedded categorical columns by:

```
dvc repro single_report
dvc repro single_column_experiments
dvc repro multi_report
```

### Caveats
- indirect testing the autoembedding model by running the outlier detections; a dedicated model evaluation not implemented yet
- normalization of continuous columns based on manual min-max scaling (`train_autoembedder.normalise_numerical_input_columns`)
- various parameters hardcoded as described above
- hyperparameters must be tuned


### Sources

Further reading and inspiration can be found in:

- https://medium.com/kirey-group/autoembedder-training-embedding-layers-on-unsupervised-tasks-fc364c0f6eec
- https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/
- https://github.com/keras-team/keras/issues/3110
- https://developers.googleblog.com/2017/11/introducing-tensorflow-feature-columns.html