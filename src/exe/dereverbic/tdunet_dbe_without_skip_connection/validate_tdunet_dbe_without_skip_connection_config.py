import pathlib as _pathlib
import torch as _torch

from exe.data.preprocess import split_dataset_config as _split_dataset_config
from exe.dereverbic.tdunet_dbe_without_skip_connection import train_tdunet_dbe_without_skip_connection_config as _train_tdunet_dbe_without_skip_connection_config


device: _torch.device = \
    _train_tdunet_dbe_without_skip_connection_config.device


checkpoints_directory: _pathlib.Path = \
    _train_tdunet_dbe_without_skip_connection_config.checkpoints_directory


start_checkpoint: int = \
    10000


validation_list: _pathlib.Path = \
    _split_dataset_config.validation_list
