import pathlib as _pathlib
import typing as _typing

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.real_world.sriracha import download_sriracha_config as _download_sriracha_config


dataset: _pathlib.Path = \
    _download_sriracha_config.destination / next(iter(_download_sriracha_config.source))


original_test_list: _pathlib.Path = \
    _split_dataset_config.test_list


output_directory: _pathlib.Path = \
    _download_sriracha_config.destination / "processed"


random_seed: str = \
    "B780F9AB-D212-4721-B925-D7BE8F955599"
