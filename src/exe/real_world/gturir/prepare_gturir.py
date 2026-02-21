from pathlib import Path
import pickle
from random import Random
import csdir
import csfile
import librosa
import numpy

import torch

from audio_processors.rir_convolution import RirConvolution


def main():
    from exe.real_world.gturir import prepare_gturir_config as config

    random: Random = Random(config.random_seed)
    output_directory: Path = csdir.create_directory(config.output_directory.absolute())

    reverb_paths: list[str] = csfile.read_all_lines(config.original_test_list)

    with open(config.dataset, "rb") as file:
        dataset: list = pickle.load(file)

    rirs: list[tuple[int, float, numpy.ndarray]]  = [
        (i, 
         (data[39] / 100) * (data[40] / 100) * (data[41] / 100), 
         librosa.resample(data[43][:44100], orig_sr=44100, target_sr=16000)) 
        for i, data in enumerate(dataset)]
    rirs = random.sample(rirs, min(len(rirs), len(reverb_paths)))

    test_list: list[str] = []

    rir_information: tuple[int, float, numpy.ndarray]
    reverb_path: str
    for rir_information, reverb_path in zip(rirs, reverb_paths):
        output_path = output_directory / Path(reverb_path).name

        print(f"{rir_information[0]} + {reverb_path} -> {output_path}")

        reverb: dict = torch.load(reverb_path, weights_only=True)
        rir: numpy.ndarray = rir_information[2]
        volume: float = rir_information[1]
        
        rir_tensor = torch.tensor(rir, dtype=torch.float)
        assert rir_tensor.shape == (16000,)
        reverb["rir"] = rir_tensor
        reverb["volume"] = volume
        reverb["reverb"] = RirConvolution.get_reverb(reverb["speech"], rir_tensor)

        torch.save(reverb, output_path)
        
        csfile.write_all_text(output_path.with_name(f"{output_path.name}.rir_path.txt"), 
                              str(rir_information[0]))
        test_list.append(str(output_path))

    csfile.write_all_lines(output_directory / "test_list.txt", test_list)
    print(f"Completed.")
    
main()