import pathlib as _pathlib
import typing as _typing

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.real_world.homularir import download_homularir_config as _download_homularir_config


inputs: _typing.Iterable[_pathlib.Path] = \
    _download_homularir_config.destination.glob("**/*.wav")


original_test_list: _pathlib.Path = \
    _split_dataset_config.test_list


output_directory: _pathlib.Path = \
    _download_homularir_config.destination / "processed"


random_seed: str = \
    "E4D23637-7577-4F02-99AB-1145867B5F26"
