"""This script provides the infrastructure to handle features in the framework.
Columns are transformed to a Feature object which carries the column's name,
flags and actions. The FeatureHandler provides the interface to all available
Features.

Usage (for testing): >> python feature_handler.py ../data/feathered
"""

import os
import json
from dataclasses import dataclass, field

import pandas as pd

from utils.utils import create_output_dir, get_dtype_dict

OUTPUT_DIRECTORY = "feature_handling"


def get_column_names_of_type(df: pd.DataFrame, type: str) -> list[str]:
    return get_dtype_dict(df)[type]


def get_names_constant_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if df[col].nunique() == 1]


def get_names_index_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if ("index" in col)]


def available_features_in_reference(
    selected_features: list[str], reference_feature_list: list[str] = []
) -> list[str]:
    """From a given list of selected feature names, return only those which are
    present in a reference list. If no reference list is given, return selected
    feature list."""
    if len(reference_feature_list) < 1:
        return sorted(selected_features)

    available_selected_features = set()
    for ref_feat in reference_feature_list:
        for feat in selected_features:
            if ref_feat.startswith(feat):
                available_selected_features.add(feat)
    return sorted(list(available_selected_features))


@dataclass
class Feature:
    """This structure stores a Feature's properties, defined as follows:

    - name: The feature's name defined either by the column name or user input.
    - is_numerical: automatically inferred using pd.api.types.is_numeric_dtype
    - is_categorical: the opposite of the above, eg. columns with string values
    - is_continuous: numerical features which have continuous values, rather than discrete
    - is_discrete: the opposite of the above, eg. numerical categories
    - drop: if the feature has been dropped
    - action: which processing step to take for the given feature
    """

    name: str = ""
    is_numerical: bool = None
    is_non_numerical: bool = None
    is_constant: bool = None
    is_index: bool = None
    actions: list[str] = field(default_factory=lambda: [])
    mapping: list = None


class FeatureHandler:
    def __init__(
        self,
        feature_names: list[str] = None,
        feature_list: list[Feature] = None,
        feature_actions: dict[str, list[str]] = {},
    ) -> None:
        """This class holds all defined features and takes care of reading them in, writing them to disk and setting various flags.

        Args:
            feature_names (list[str], optional): List of all feature names as defined by column name or user. Defaults to None.
            feature_list (list[str], optional): List of Feature instances. Defaults to None.
        """
        self._feature_names = feature_names
        self._feature_list = feature_list
        self._feature_actions = feature_actions

        self._processed_user_actions = False

    def __str__(self) -> str:
        sorted_feat_list = sorted(self._feature_list, key=lambda feat: feat.name)
        features = json.dumps(sorted_feat_list, default=lambda o: o.__dict__, indent=4)
        return f"FeatureHandler._feature_list:\n{features}"

    def to_json(self, out_path: str) -> None:
        """Write this instance to json, such that it can be re-instatiated from file later."""
        create_output_dir(out_path)
        with open(os.path.join(out_path, "feature_handler.json"), "w") as file:
            json.dump(self, file, default=lambda o: o.__dict__, indent=4)

    def pretty_print_to_json(self, out_path: str) -> None:
        """Write a human-readable, concise summary to disk for user review only."""
        create_output_dir(out_path)
        with open(
            os.path.join(out_path, "feature_handler_pretty_summary.json"), "w"
        ) as file:
            print_summary = {
                "all features": [self._feature_names],
                "numerical features": [
                    feat.name for feat in self._feature_list if feat.is_numerical
                ],
                "is_non_numerical features": [
                    feat.name for feat in self._feature_list if feat.is_non_numerical
                ],
                "is_constant features": [
                    feat.name for feat in self._feature_list if feat.is_constant
                ],
                "is_index features": [
                    feat.name for feat in self._feature_list if feat.is_index
                ],
                "auto_embedded_features": [
                    feat.name for feat in self._feature_list if "_emb_" in feat.name
                ],
            }
            json.dump(print_summary, file, indent=4)

    @property
    def processed_user_actions(self) -> bool:
        return self._processed_user_actions

    @classmethod
    def from_json(cls, input_file: str):
        """Create the FeatureHandler instance from a yaml file containing its members. This is useful to re-instantiate after saving to disk.

        Args:
            input_dir (str): Input directory of json

        Returns:
            FeatureHandler: Instance of the FeatureHandler class.
        """
        with open(f"{input_file}", "r") as in_file:
            handler_specs = json.load(in_file)
        feature_names = handler_specs["_feature_names"]
        feature_list = [
            Feature(**feat_specs) for feat_specs in handler_specs["_feature_list"]
        ]
        return cls(feature_names=feature_names, feature_list=feature_list)

    @classmethod
    def from_df(
        cls,
        df: pd.DataFrame,
        feature_actions: dict[str, list[str]] = {},
    ):
        """Create the feature handler from a given dataframe. Used when loading
        the feature set for the first time.

        Args:
            df (pd.DataFrame): single input report

        Returns:
            FeatureHandler: Instance of the FeatureHandler class.
        """
        column_list = list(df.columns)
        instance = cls(feature_names=column_list, feature_actions=feature_actions)
        instance._create_feature_list()
        instance._set_feature_flags(df, instance._feature_list)
        instance._set_user_actions()
        return instance

    def _create_feature_list(self) -> None:
        self._feature_list = self._create_feature_instances_from_column_names(
            self._feature_names
        )

    def _create_feature_instances_from_column_names(self, column_list) -> list[Feature]:
        return [Feature(feat_name) for feat_name in column_list]

    def _set_user_actions(self) -> None:
        for feat in self._feature_list:
            feat.actions = [
                action
                for action, feat_name_ref in self._feature_actions.items()
                if feat.name in feat_name_ref
            ]

    def set_user_actions(self) -> None:
        self._set_user_actions()
        self._processed_user_actions == True

    # TODO
    # def sanity_checks():
    #   all features have an action?

    def _get_feature_with_name(self, feat_name: str) -> Feature:
        return [feat for feat in self._feature_list if feat_name == feat.name][0]

    def add_mappings_to_features(self, mappings: dict) -> None:
        for feat_name, mapping in mappings.items():
            this_feat = self._get_feature_with_name(feat_name)
            this_feat.mapping = list(mapping[0])

    def _set_feature_flags(self, df: pd.DataFrame, feature_list: list[Feature]) -> None:
        for feat in feature_list:
            feat.is_numerical = feat.name in get_column_names_of_type(df, "numerical")
            feat.is_non_numerical = feat.name in get_column_names_of_type(
                df, "categorical"
            )
            feat.is_constant = feat.name in get_names_constant_columns(df)
            feat.is_index = feat.name in get_names_index_columns(df)

    def add_features_from_df(self, df: pd.DataFrame) -> None:
        new_features = self._create_feature_instances_from_column_names(df.columns)
        for feat in new_features:
            if feat.name not in self._feature_names:
                self._feature_names.append(feat.name)
                self._feature_list.append(
                    Feature(
                        name=feat.name,
                        is_numerical=feat.name
                        in get_column_names_of_type(df, "numerical"),
                        is_non_numerical=feat.name
                        in get_column_names_of_type(df, "categorical"),
                        is_constant=feat.name in get_names_constant_columns(df),
                        is_index=feat.name in get_names_index_columns(df),
                    )
                )

    def get_selected_features(
        self,
        include_features_with_flags: list[str] = [],
        include_features_with_actions: list[str] = [],
        exclude_features_with_flags: list[str] = [],
        exclude_features_with_actions: list[str] = [],
    ) -> list[Feature]:
        """This method returns lists of Feature instances according to the
        specified flags. Features are considered, if:
        - all flags in include_features_with_flags are true

        Args:
            include_features_with_flags (list[str]): list of criteria a given feature has to fulfill, eg. ["is_numerical"]

        Raises:
            ValueError: if a criterion in include_features_with_flags/exclude_features_with_flags is not set for a given feature

        Returns:
            list[Feature]: List of selected features.
        """
        selected_features = []
        for feat in self._feature_list:
            for incl_crit in include_features_with_flags:
                if getattr(feat, incl_crit) is None:
                    raise ValueError(
                        f"Requested attribute {incl_crit} for {feat.name} is None; has not been set."
                    )
                if getattr(feat, incl_crit):
                    selected_features.append(feat)
            for incl_action in include_features_with_actions:
                if incl_action in feat.actions:
                    selected_features.append(feat)
        if (
            len(exclude_features_with_flags) == 0
            or len(exclude_features_with_actions) == 0
        ):
            return selected_features
        else:
            selected_features_w_exclusion = (
                selected_features if len(selected_features) > 0 else self._feature_list
            )
            for excl_flag in exclude_features_with_flags:
                selected_features_w_exclusion = [
                    sel_feat
                    for sel_feat in selected_features_w_exclusion
                    if not getattr(sel_feat, excl_flag)
                ]
            for excl_action in exclude_features_with_actions:
                selected_features_w_exclusion = [
                    sel_feat
                    for sel_feat in selected_features_w_exclusion
                    if not excl_action in sel_feat.actions
                ]
            return selected_features_w_exclusion

    def get_selected_features_names(
        self,
        include_features_with_flags: list[str] = [],
        include_features_with_actions: list[str] = [],
        available_feature_names: list[str] = [],
        exclude_features_with_flags: list[str] = [],
        exclude_features_with_actions: list[str] = [],
    ) -> list[str]:
        """A wrapper around get_selected_features to return the names of features."""
        selected_features_names = [
            feat.name
            for feat in self.get_selected_features(
                include_features_with_flags,
                include_features_with_actions,
                exclude_features_with_flags,
                exclude_features_with_actions,
            )
        ]
        return available_features_in_reference(
            selected_features_names, available_feature_names
        )