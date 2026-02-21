from pathlib import Path
from zipfile import ZipFile
import csdir
import urllib.request


def main():
    from exe.real_world.sriracha import download_sriracha_config as config
    destination: Path = csdir.create_directory(config.destination)
    
    name: str
    source: str
    for name, source in config.source.items():
        dataset_path: Path = destination / name

        print(f"Downloading {source} ...")
        urllib.request.urlretrieve(source, dataset_path)
        
    print(f"Completed.")
    
main()