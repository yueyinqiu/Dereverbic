import pathlib as _pathlib
from typing import Iterable, Mapping

from exe import common_configurations as _common_configurations


source: Mapping[str, str] = \
    {
        "SR1-C1.h5": "https://api-depositonce.tu-berlin.de/server/api/core/bitstreams/75b3da38-9f7f-4b30-8967-3da1b0f29e9f/content"
    }


destination: _pathlib.Path = \
    _common_configurations.data_directory / "real_world" / "sriracha"
