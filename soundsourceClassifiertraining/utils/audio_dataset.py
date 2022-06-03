"""Audio dataset class."""
# Third party modules
import torch
from torch.utils.data import Dataset

# Local modules
from soundsourceClassifiertraining.constants import SAMPLE_RATE
from soundsourceClassifiertraining.utils.processing import get_mel_spectrogram


class AudioDataset(Dataset):
    """Audio dataset"""

    def __init__(self, X_train, y_train, transform=None):
        """Init"""
        self.X_train = X_train
        self.y_train = y_train
        self.transform = transform

    def __len__(self):
        """Length of dataset"""
        return len(self.X_train)

    def __getitem__(self, index):
        """Get items"""
        signal, class_id = self.X_train[index], self.y_train[index]
        spectrogram = get_mel_spectrogram(signal, SAMPLE_RATE)
        if self.transform:
            signal = torch.unsqueeze(signal, dim=0)
            signal = self.transform(signal, SAMPLE_RATE)
            if not torch.is_tensor(signal):
                signal = torch.from_numpy(signal)
            signal = torch.squeeze(signal)
            spectrogram = get_mel_spectrogram(signal, SAMPLE_RATE)
        return spectrogram, class_id
