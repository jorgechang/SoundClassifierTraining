"""Classifiertraining Model."""
# Standar modules
import random

# Third party modules
import torch
import torchaudio
from pydantic.dataclasses import dataclass

# Local modules
from soundsourceClassifiertraining.constants import (
    SAMPLE_RATE,
    max_audio_length,
    top_db,
)


@dataclass
class Processaudio:
    """Process Audio."""

    audio_file: str

    def __post_init_post_parse__(self):
        """Post init section."""
        self.__extract_signal()
        self.__convert_two_channels()
        self.__fix_sample_rate()
        self.__pad_or_truncate()
        self.__normalize_signal()

    def __normalize_signal(self):
        signal_mean, signal_std = self.signal.mean(), self.signal.std()
        self.signal = (self.signal - signal_mean) / signal_std

    def __pad_or_truncate(self):
        num_channels, signal_len = self.signal.shape
        max_len = self.sample_rate // 1000 * max_audio_length

        # Truncate
        if signal_len > max_len:
            self.signal = self.signal[:, :max_len]
        # Pad
        elif signal_len < max_len:
            pad_begin_len = random.randint(0, max_len - signal_len)
            pad_end_len = max_len - signal_len - pad_begin_len

            pad_begin = torch.zeros((num_channels, pad_begin_len))
            pad_end = torch.zeros((num_channels, pad_end_len))

            self.signal = torch.cat((pad_begin, self.signal, pad_end), 1)

    def __fix_sample_rate(self):
        if self.sample_rate != SAMPLE_RATE:
            self.signal = torchaudio.transforms.Resample(self.sample_rate, SAMPLE_RATE)(
                self.signal[:1, :]
            )

    def __convert_two_channels(self):
        num_channels = self.signal.shape[0]
        if num_channels != 2:
            mean_signal = torch.mean(self.signal, dim=0).unsqueeze(0)
            self.signal = torch.cat((mean_signal, mean_signal), dim=0)

    def __extract_signal(self):
        self.signal, self.sample_rate = torchaudio.load(self.audio_file)


def get_mel_spectrogram(signal, sample_rate):
    """Gets mel spectogram"""
    spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate, n_fft=1024, n_mels=64
    )(signal)
    spectrogram = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spectrogram)
    return spectrogram
