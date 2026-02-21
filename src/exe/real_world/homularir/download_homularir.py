import csv
import io
from pathlib import Path
from random import Random
from zipfile import ZipFile
import csdir
from statictorch import Tensor1d, Tensor2d
import urllib.request

from basic_utilities.string_random import StringRandom
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.tensor_audios.tensor_audios import TensorAudios


def main():
    from exe.real_world.homularir import download_homularir_config as config
    source: str = config.source
    destination: Path = csdir.create_directory(config.destination)

    dataset_path: Path = destination / "HOMULA-RIR.zip"

    print(f"Downloading {source} ...")
    urllib.request.urlretrieve(source, dataset_path)
    
    print(f"Extracting {dataset_path} ...")
    zip: ZipFile
    with ZipFile(dataset_path) as zip:
        zip.extractall(destination)
    
    print(f"Completed.")
    
main()