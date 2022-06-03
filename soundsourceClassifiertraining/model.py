"""Classifiertraining Model."""
# Standar modules
import copy

# Third party modules
import torch
import torch.nn as nn
from pydantic.dataclasses import dataclass
from torch.utils.data import DataLoader

# Local modules
from soundsourceClassifiertraining.constants import PATH
from soundsourceClassifiertraining.utils.audio_augment import AudioAugment
from soundsourceClassifiertraining.utils.audio_dataset import AudioDataset
from soundsourceClassifiertraining.utils.audio_processing import AudioProcessing
from soundsourceClassifiertraining.utils.classifier import AudioClassifier


@dataclass
class Training:
    """Classifiertraining Model."""

    data_path: str
    labels: list
    transforms: list = None
    batch_size: int = 24

    def __post_init_post_parse__(self):
        """Post init section."""
        processor = AudioProcessing(self.data_path, self.labels)
        (
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
        ) = processor.get_splitted_data()
        self.train_dataset = AudioDataset(self.X_train, self.y_train)
        if self.transforms:
            augmentations = AudioAugment(self.X_train, self.y_train, self.transforms)
            self.train_dataset = augmentations.apply_transforms(self.train_dataset)
        self.train_dl = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_dataset = AudioDataset(self.X_test, self.y_test)
        self.val_dl = DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False
        )

        self.__load_model()

    def __str__(self):
        """Model docstring."""
        pass

    def __load_model(self) -> None:
        """Load AudioClassifier model."""
        self.model = AudioClassifier(len(self.labels))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def eval_model(self):
        """Inference on the test set"""
        correct_prediction = 0
        total_prediction = 0

        for data in self.val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # Normalize the inputs
            inputs_mean, inputs_std = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_mean) / inputs_std

            # Get predictions
            outputs = self.model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

        acc = correct_prediction / total_prediction
        print(f"Accuracy: {acc:.2f}, Total items: {total_prediction}")

    def train_model(self, num_epochs: int):
        """Train the model

        Parameters:
        -----------
            num_epochs [int]:
                Number of epochs to train the model.

        Returns:
        --------
            Tuple[float, float]:
                Tuple containing the accuracy and loss.
        """
        num_batches = len(self.train_dl)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=0.001,
            steps_per_epoch=int(len(self.train_dl)),
            epochs=num_epochs,
            anneal_strategy="linear",
        )
        best_acc = 0.0

        # Repeat for each epoch
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct_prediction = 0
            total_prediction = 0

            best_model_wts = copy.deepcopy(self.model.state_dict())
            for inputs, labels in self.train_dl:
                # Get the input features and target labels, and put them on the GPU
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Normalize the inputs
                inputs_mean, inputs_std = inputs.mean(), inputs.std()
                inputs = (inputs - inputs_mean) / inputs_std

                # Zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Keep stats for Loss and Accuracy
                running_loss += loss.item()
                # Get the predicted class with the highest score
                _, prediction = torch.max(outputs, 1)
                # Count of predictions that matched the target label
                correct_prediction += (prediction == labels).sum().item()
                total_prediction += prediction.shape[0]

                # Print stats at the end of the epoch
                avg_loss = running_loss / num_batches
                acc = correct_prediction / total_prediction

            if acc > best_acc:
                best_acc = acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
            print(f"Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}")

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), PATH)
        print(f"Model saved at {PATH}")
