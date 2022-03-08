"""Utilities functions for parameters reading."""

import yaml


def load_config_subsection(filename: str, section: str) -> dict:
    with open(filename, "r") as conf_file:
        conf = yaml.safe_load(conf_file)

    return extract_subsections(conf, section)


def extract_subsections(conf: dict, sections: list[str]) -> dict:
    """Extracts sections from a nested dictionary, either using a single key or a list of keys.

    >>> test_dict = {'foo': {'bar': {'a': 42, 'b': 55}}, 'baz': "Hello"}
    >>> extract_subsection(test_dict, ["foo.bar","baz"])
    {foo.bar.a: 42, foo.bar.b: 55, baz: "Hello"}
    """

    if isinstance(sections, list):
        config_dict = {}
        for section in sections:
            subsection = extract_subsection(conf, section)
            if isinstance(subsection, dict):
                config_dict.update(
                    {
                        f"{section}.{parameter}": value
                        for parameter, value in subsection.items()
                    }
                )
            else:
                config_dict[section] = subsection

        return config_dict
    else:
        return extract_subsection(conf, sections)


def extract_subsection(conf: dict, section: str) -> dict:
    """Takes a nested dictionary and returns the dictionary corresponding to a dot-delimited sub-dict.

    >>> test_dict = {'foo': {'bar': {1: 42, 2: 55}}, 'baz': "Hello"}
    >>> extract_subsection(test_dict, "foo.bar")
    {1: 42, 2: 55}
    """

    for subsection in section.split("."):
        conf = conf[subsection]
    return conf


def with_params(filename: str, section: str):
    def decorator(function):
        def wrapper(*args, **kwargs):
            param_dict = load_config_subsection(filename, section)
            return function(*args, **kwargs, params=param_dict)

        return wrapper

    return decorator
