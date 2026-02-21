from pathlib import Path
from random import Random
import csdir
import csfile
import librosa
import numpy

import torch

from audio_processors.rir_convolution import RirConvolution


def main():
    from exe.real_world.homularir import prepare_homularir_config as config

    random: Random = Random(config.random_seed)
    output_directory: Path = csdir.create_directory(config.output_directory.absolute())

    reverb_paths: list[str] = csfile.read_all_lines(config.original_test_list)

    rir_paths: list[tuple[Path, int]]  = [
        (path.absolute(), channel) 
        for path in sorted(config.inputs)
        for channel in range(librosa.load(path, sr=16000, mono=False)[0].shape[0])]
    rir_paths = random.sample(rir_paths, min(len(rir_paths), len(reverb_paths)))

    test_list: list[str] = []

    rir_path: tuple[Path, int]
    reverb_path: str
    for rir_path, reverb_path in zip(rir_paths, reverb_paths):
        output_path = output_directory / Path(reverb_path).name

        print(f"{rir_path[0]} ({rir_path[1]}) + {reverb_path} -> {output_path}")

        reverb: dict = torch.load(reverb_path, weights_only=True)
        rir: numpy.ndarray
        rir, _ = librosa.load(rir_path[0], sr=16000, mono=False)
        rir = rir[rir_path[1]]
        
        rir_tensor = torch.tensor(rir, dtype=torch.float)
        assert rir_tensor.shape == (16000,)
        reverb["rir"] = rir_tensor
        reverb["volume"] = 14.52 * 5.46 * 3.38
        reverb["reverb"] = RirConvolution.get_reverb(reverb["speech"], rir_tensor)

        torch.save(reverb, output_path)
        
        csfile.write_all_lines(output_path.with_name(f"{output_path.name}.rir_path.txt"), 
                               (str(rir_path[0]), str(rir_path[1])))
        test_list.append(str(output_path))

    csfile.write_all_lines(output_directory / "test_list.txt", test_list)
    print(f"Completed.")
    
main()