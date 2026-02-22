import csv
import io
from pathlib import Path
import sys
from typing import Any, Iterable, TypedDict
import csfile
from statictorch import Tensor1d, Tensor2d, anify
import torch

from audio_processors.rir_acoustic_features import RirAcousticFeatures2d
from inputs_and_outputs.checkpoint_managers.checkpoints_directory import CheckpointsDirectory
from inputs_and_outputs.checkpoint_managers.epoch_and_path import EpochAndPath
from inputs_and_outputs.csv_accessors.csv_reader import CsvReader
from inputs_and_outputs.csv_accessors.csv_writer import CsvWriter
from inputs_and_outputs.data_providers.data_batch import DataBatch
from inputs_and_outputs.data_providers.validation_or_test_dataset import ValidationOrTestDataset
from metrics.bias_metric import BiasMetric
from metrics.l1_loss_metric import L1LossMetric
from metrics.metric import Metric
from metrics.mrstft_loss_metric import MrstftLossMetric
from metrics.pearson_correlation_metric import PearsonCorrelationMetric
from metrics.rir_direct_to_reverberant_energy_ratio_metrics import RirDirectToReverberantEnergyRatioMetrics
from metrics.rir_reverberation_time_metrics import RirReverberationTimeMetrics
from models.berp_models.berp_hybrid_model import BerpHybridModel
from trainers.trainer import Trainer

class _TestDatasetWithVolume(torch.utils.data.Dataset):
    def __init__(self, 
                 data_list: Path, 
                 device: torch.device):
        self._paths = csfile.read_all_lines(data_list)
        self._device = device

    def __len__(self):
        return self._paths.__len__()

    class DatasetItem(TypedDict):
        rir: Tensor1d
        speech: Tensor1d
        reverb: Tensor1d

    def __getitem__(self, i: int) -> tuple[DatasetItem, float]:
        path: str = self._paths[i]
        result: Any = torch.load(path, weights_only=True, map_location=self._device)
        return result, result["volume"]

    def get_data_loader(self, batch_size: int) -> torch.utils.data.DataLoader:
        def collate(data: list[tuple[ValidationOrTestDataset.DatasetItem, float]]) -> tuple[DataBatch, Tensor1d]:
            rirs: list[Tensor1d] = []
            speeches: list[Tensor1d] = []
            reverbs: list[Tensor1d] = []
            volumes: list[float] = []

            item: ValidationOrTestDataset.DatasetItem
            volume: float
            for item, volume in data:
                rirs.append(item["rir"])
                speeches.append(item["speech"])
                reverbs.append(item["reverb"])
                volumes.append(volume)
            
            result: DataBatch = DataBatch(Tensor2d(torch.stack(anify(rirs))), 
                                          Tensor2d(torch.stack(anify(speeches))), 
                                          Tensor2d(torch.stack(anify(reverbs))))
            return result, Tensor1d(torch.tensor(volumes, device=self._device))

        return torch.utils.data.DataLoader(self, batch_size, False, collate_fn=collate)


def test(model: BerpHybridModel, 
         checkpoints: CheckpointsDirectory,
         data: torch.utils.data.DataLoader, 
         rir_metrics: dict[str, Metric[Tensor2d]],
         feature_metrics: dict[str, Metric[RirAcousticFeatures2d]]):
    with torch.no_grad():
        print(f"# Batch count: {data.__len__()}")

        rank_file: Path = checkpoints.get_path(None) / "validation_rank.txt"
        if rank_file.exists():
            epoch: int = int(csfile.read_all_lines(rank_file)[0])
            path: Path = checkpoints.get_path(epoch)
            print(f"# Rank file found. The best checkpoint {epoch} will be used.")
        else:
            latest: EpochAndPath | None = checkpoints.get_latest()
            if not latest:
                raise FileNotFoundError("Failed to find any checkpoint in the checkpoints directory.")
            epoch, path = latest
            print(f"# Failed to find the rank file. The latest checkpoint {epoch} will be used.")

        csv_print: CsvWriter = csv.writer(sys.stdout)
        csv_print.writerow(["batch", "metric", "submetric", "value"])

        Trainer.load_model(model, path)

        batch_index: int
        batch: tuple[DataBatch, Tensor1d]
        for batch_index, batch in enumerate(data):
            predicted: Tensor2d = model.evaluate_rir_on(batch[0].reverb, batch[1])

            metric: str
            for metric in rir_metrics:
                current: dict[str, float] = rir_metrics[metric].append(batch[0].rir, predicted)
                submetric: str
                for submetric in current:
                    csv_print.writerow([batch_index, metric, submetric, current[submetric]])

            actual_features: RirAcousticFeatures2d = RirAcousticFeatures2d(batch[0].rir)
            predicted_features: RirAcousticFeatures2d = RirAcousticFeatures2d(predicted)
            for metric in feature_metrics:
                current = feature_metrics[metric].append(actual_features, predicted_features)
                for submetric in current:
                    csv_print.writerow([batch_index, metric, submetric, current[submetric]])

        for metric in rir_metrics:
            value: float
            for submetric, value in rir_metrics[metric].result().items():
                csv_print.writerow(["all", metric, submetric, value])

        for metric in feature_metrics:
            for submetric, value in feature_metrics[metric].result().items():
                csv_print.writerow(["all", metric, submetric, value])


def main():
    from exe.real_world.gturir import test_berp_on_gturir_config as config
    
    data: _TestDatasetWithVolume = _TestDatasetWithVolume(config.test_list,
                                                          config.device)
    test(BerpHybridModel(config.device),
         CheckpointsDirectory(config.checkpoints_directory),
         data.get_data_loader(32),
         {
             "mrstft": MrstftLossMetric.for_rir(config.device)
         },
         {
             "rt60": RirReverberationTimeMetrics(30, 16000, {
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(config.device),
                 "pearson": PearsonCorrelationMetric()
             }),
             "drr": RirDirectToReverberantEnergyRatioMetrics({
                 "bias": BiasMetric(),
                 "l1": L1LossMetric(config.device),
                 "pearson": PearsonCorrelationMetric()
             }),
         })


if __name__ == "__main__":
    main()