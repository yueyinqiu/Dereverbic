import pathlib as _pathlib

from exe import common_configurations as _common_configurations


source: str = \
    "https://zenodo.org/records/15053008/files/HOMULA-RIR.zip?download=1"


destination: _pathlib.Path = \
    _common_configurations.data_directory / "real_world" / "homularir"
