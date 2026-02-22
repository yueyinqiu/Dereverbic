import pathlib as _pathlib
import torch as _torch

from exe.dereverbic.dereverbic import test_dereverbic_config as _test_dereverbic_config
from exe.real_world.gturir import prepare_gturir_config as _prepare_gturir_config


device: _torch.device = \
    _test_dereverbic_config.device


checkpoints_directory: _pathlib.Path = \
    _test_dereverbic_config.checkpoints_directory


test_list: _pathlib.Path = \
    _prepare_gturir_config.output_directory / "test_list.txt"
