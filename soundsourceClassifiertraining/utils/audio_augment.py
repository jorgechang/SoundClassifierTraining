"""Audio dataset class."""
# Third party modules
from audiomentations import AddGaussianNoise, Compose, FrequencyMask
from pydantic.dataclasses import dataclass
from torch.utils.data import ConcatDataset
from torch_audiomentations import Gain, Shift

# Local modules
from soundsourceClassifiertraining.utils.audio_airabsorption import AirAbsorption
from soundsourceClassifiertraining.utils.audio_dataset import AudioDataset


@dataclass
class AudioAugment:
    """Audio dataset"""

    X_train: list
    y_train: list
    transforms: list

    def __post_init_post_parse__(self):
        """Init augmentation fucntions"""
        self.functions = {
            "AirAbsorption": Compose([AirAbsorption()]),
            "Gain": Compose([Gain(min_gain_in_db=-15.0, max_gain_in_db=5.0, p=0.9)]),
            "GaussianNoise": Compose(
                [AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.9)]
            ),
            "FrequencyMask": Compose([FrequencyMask()]),
            "Shift": Compose([Shift()]),
        }

    def apply_transforms(self, train_dataset):
        """Apply chosen transforms"""
        datasets = [train_dataset]

        for transform in self.transforms:

            augmented_dataset = AudioDataset(
                self.X_train, self.y_train, self.functions[transform]
            )
            datasets.append(augmented_dataset)

        train_augmented_dataset = ConcatDataset(datasets)
        return train_augmented_dataset
