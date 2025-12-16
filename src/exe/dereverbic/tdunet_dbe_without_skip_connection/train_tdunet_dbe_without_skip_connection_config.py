import pathlib as _pathlib
import torch as _torch

from exe import common_configurations as _common_configurations
from exe.data.preprocess import split_dataset_config as _split_dataset_config


device: _torch.device = \
    _common_configurations.device


checkpoint_interval: int = \
    1000


checkpoints_directory: _pathlib.Path = \
    _common_configurations.checkpoints_directory / "tdunet_dbe_without_skip_connection/"


train_list_rir: _pathlib.Path = \
    _split_dataset_config.train_list_rir


train_list_speech: _pathlib.Path = \
    _split_dataset_config.train_list_speech


random_seed: str = \
    "F87F1A32-E69A-4029-B51B-3774DA17C0D8"
