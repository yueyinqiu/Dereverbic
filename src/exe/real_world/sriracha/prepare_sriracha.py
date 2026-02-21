from pathlib import Path
from random import Random
import csdir
import csfile
import h5py
import librosa
import numpy

import torch

from audio_processors.rir_convolution import RirConvolution


def main():
    from exe.real_world.sriracha import prepare_sriracha_config as config

    random: Random = Random(config.random_seed)
    output_directory: Path = csdir.create_directory(config.output_directory.absolute())

    reverb_paths: list[str] = csfile.read_all_lines(config.original_test_list)

    with h5py.File(config.dataset) as file:
        rir_array: numpy.ndarray = file["data"]["impulse_response"][()]    # type: ignore

    rir_index_list: list[tuple[int, int]] = [
        (i, j)
        for i in range(rir_array.shape[0])
        for j in range(rir_array.shape[1])
    ]
    rir_index_list = random.sample(rir_index_list, min(len(rir_index_list), len(reverb_paths)))

    test_list: list[str] = []
    
    rir_index: tuple[int, int]
    reverb_path: str
    for rir_index, reverb_path in zip(rir_index_list, reverb_paths):
        output_path = output_directory / Path(reverb_path).name

        print(f"{rir_index[0]} + {reverb_path} -> {output_path}")

        reverb: dict = torch.load(reverb_path, weights_only=True)
        rir: numpy.ndarray = rir_array[rir_index[0], rir_index[1], ...]
        rir = librosa.resample(rir, orig_sr=32000, target_sr=16000)
        
        rir_tensor = torch.tensor(rir, dtype=torch.float)
        assert rir_tensor.shape == (16000,)
        reverb["rir"] = rir_tensor
        reverb["volume"] = 6.22 * 3.85 * 3.07
        reverb["reverb"] = RirConvolution.get_reverb(reverb["speech"], rir_tensor)

        torch.save(reverb, output_path)
        
        csfile.write_all_text(output_path.with_name(f"{output_path.name}.rir_path.txt"), str(rir_index))
        test_list.append(str(output_path))

    csfile.write_all_lines(output_directory / "test_list.txt", test_list)
    print(f"Completed.")
    
main()