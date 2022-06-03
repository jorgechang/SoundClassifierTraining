"""Audio processing functions."""
# Standar modules
import os

# Third party modules
import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.model_selection import train_test_split

# Local modules
from soundsourceClassifiertraining.utils.processing import Processaudio


@dataclass
class AudioProcessing:
    """Process Audio."""

    data_path: str
    training_labels: list

    def __post_init_post_parse__(self):
        """Post init section."""
        self.__extract_dataframe()
        self.__audiodataset()

    def __extract_dataframe(self):
        """Extracts audio signal from dataframe."""
        meta_path = self.data_path + "/FSD50K.metadata/collection"

        meta_dataframe = pd.read_csv(meta_path + "/collection_dev.csv")
        voc_dataframe = pd.read_csv(
            meta_path + "/collection_eval.csv", names=["classID", "Label", "mid"]
        )

        df = meta_dataframe[meta_dataframe["labels"].isin(self.training_labels)]
        df["fname"] = df["fname"].apply(lambda x: str(x) + ".wav")
        df["classID"] = df["labels"].apply(
            lambda x: voc_dataframe[voc_dataframe["Label"] == x]["classID"].values[0]
        )
        df = df.reset_index(drop=True)
        labels = df["classID"].unique().tolist()
        df["Labels"] = df["classID"].apply(lambda x: labels.index(x))
        self.dataframe = df

    def __audiodataset(self):
        audio_path = self.data_path + "/FSD50K.dev_audio"
        self.signals = []
        self.sample_rate_s = 0
        self.labels = []
        for data in self.dataframe.index:
            audio_file = os.path.join(audio_path, self.dataframe.iloc[data]["fname"])
            class_id = self.dataframe.iloc[data]["Labels"]
            audio = Processaudio(audio_file)
            self.signals.append(audio.signal)
            self.labels.append(class_id)
            self.sample_rate_s = audio.sample_rate

    def get_splitted_data(self):
        """Get splitted data."""
        return train_test_split(self.signals, self.labels, test_size=0.3)
