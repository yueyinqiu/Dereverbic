import pathlib as _pathlib

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe import common_configurations as _common_configurations


# This file is published on OneDrive alone, so it may need to be downloaded manually; we failed to write a automated script.
# See https://github.com/Graphi07/room-impulse-responses
dataset: _pathlib.Path = \
    _common_configurations.data_directory / "real_world" / "gturir" / "RIR.pickle.dat"


original_test_list: _pathlib.Path = \
    _split_dataset_config.test_list


output_directory: _pathlib.Path = \
    _common_configurations.data_directory / "real_world" / "gturir" / "processed"


random_seed: str = \
    "2944BA21-2BDC-4192-9EA6-CFC0BBCC20A4"
