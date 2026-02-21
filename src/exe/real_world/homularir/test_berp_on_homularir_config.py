import pathlib as _pathlib
import torch as _torch

from exe.berp import test_berp_config as _test_berp_config
from exe.real_world.homularir import prepare_homularir_config as _prepare_homularir_config


device: _torch.device = \
    _test_berp_config.device


checkpoints_directory: _pathlib.Path = \
    _test_berp_config.checkpoints_directory


test_list: _pathlib.Path = \
    _prepare_homularir_config.output_directory / "test_list.txt"
