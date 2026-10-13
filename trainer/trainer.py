from os import PathLike
from pathlib import Path
from typing import Sequence

import torch
import tqdm
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from trainer.metrics import (
    EpochMetrics,
    calculate_metrics,
    format_classification_table,
)


class TETrainer:
    """Train and evaluate VariantSeqClassify with MEIDataset batches."""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader | None = None,
        with_cuda: bool = True,
        lr: float = 1e-6,
        cuda_devices: Sequence[int] | None = None,
        log_freq: int = 10,
        save_path: str | PathLike[str] = "fine_tuned_model",
    ) -> None:
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        self.model = model.to(self.device)
        if cuda_condition and cuda_devices is not None and len(cuda_devices) > 1:
            self.model = nn.DataParallel(self.model, device_ids=list(cuda_devices))

        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.optim = Adam(self.model.parameters(), lr=lr)
        self.log_freq = log_freq
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)

        parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        print("Device:", self.device)
        print("Total Parameters:", parameter_count)

    def train(self, num_epochs: int) -> list[EpochMetrics]:
        """Train the model and save one checkpoint after every epoch."""
        if num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero")
        if len(self.train_data) == 0:
            raise RuntimeError("train_dataloader must contain at least one batch")

        history: list[EpochMetrics] = []

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            all_predictions: list[int] = []
            all_labels: list[int] = []
            data_iter = tqdm.tqdm(
                enumerate(self.train_data),
                desc=f"EP:{epoch + 1}",
                total=len(self.train_data),
                bar_format="{l_bar}{r_bar}",
                position=0,
            )

            for batch_index, (sequences, one_hot_labels) in data_iter:
                sequences = sequences.to(self.device)
                one_hot_labels = one_hot_labels.to(self.device)
                label_ids = torch.argmax(one_hot_labels, dim=1)

                self.optim.zero_grad()
                probabilities = self.model(sequences)
                loss = self._loss(probabilities, one_hot_labels)
                loss.backward()
                self.optim.step()

                predictions = torch.argmax(probabilities, dim=1)
                batch_size = label_ids.shape[0]
                total_loss += loss.item() * batch_size
                total_correct += (predictions == label_ids).sum().item()
                all_predictions.extend(predictions.detach().cpu().tolist())
                all_labels.extend(label_ids.detach().cpu().tolist())

                if batch_index % self.log_freq == 0:
                    data_iter.write(
                        str(
                            calculate_metrics(
                                total_loss,
                                total_correct,
                                all_labels,
                                all_predictions,
                            )
                        )
                    )

            metrics = calculate_metrics(
                total_loss,
                total_correct,
                all_labels,
                all_predictions,
            )
            history.append(metrics)
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train_loss={metrics.loss:.6f} | "
                f"train_acc={metrics.accuracy:.2f}% | "
                f"train_f1={metrics.f1_score:.4f} | "
                f"lr={self.optim.param_groups[0]['lr']:.8f}"
            )
            self._save_checkpoint(epoch + 1, metrics)

        return history

    def test(self) -> EpochMetrics:
        """Evaluate the current model on the configured test dataloader."""
        if self.test_data is None:
            raise RuntimeError("test_dataloader is required before calling test()")
        metrics = self._evaluate(print_table=True)
        if metrics is None:
            raise RuntimeError("test_dataloader must contain at least one batch")
        return metrics

    def _evaluate(self, print_table: bool) -> EpochMetrics | None:
        if self.test_data is None:
            return None
        if len(self.test_data) == 0:
            return None
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        all_predictions: list[int] = []
        all_labels: list[int] = []
        data_iter = tqdm.tqdm(
            enumerate(self.test_data),
            desc="TEST",
            total=len(self.test_data),
            bar_format="{l_bar}{r_bar}",
            position=0,
        )

        with torch.no_grad():
            for batch_index, (sequences, one_hot_labels) in data_iter:
                sequences = sequences.to(self.device)
                one_hot_labels = one_hot_labels.to(self.device)
                label_ids = torch.argmax(one_hot_labels, dim=1)
                probabilities = self.model(sequences)
                loss = self._loss(probabilities, one_hot_labels)
                predictions = torch.argmax(probabilities, dim=1)

                batch_size = label_ids.shape[0]
                total_loss += loss.item() * batch_size
                total_correct += (predictions == label_ids).sum().item()
                all_predictions.extend(predictions.cpu().tolist())
                all_labels.extend(label_ids.cpu().tolist())

                if batch_index % self.log_freq == 0:
                    data_iter.write(
                        str(
                            calculate_metrics(
                                total_loss,
                                total_correct,
                                all_labels,
                                all_predictions,
                            )
                        )
                    )

        metrics = calculate_metrics(
            total_loss,
            total_correct,
            all_labels,
            all_predictions,
        )
        print("TEST", metrics)
        if print_table:
            print(format_classification_table(all_labels, all_predictions))
        return metrics

    @staticmethod
    def _loss(probabilities: Tensor, one_hot_labels: Tensor) -> Tensor:
        minimum = torch.finfo(probabilities.dtype).tiny
        log_probabilities = torch.log(probabilities.clamp_min(minimum))
        return -(one_hot_labels * log_probabilities).sum(dim=1).mean()

    def _save_checkpoint(self, epoch: int, metrics: EpochMetrics) -> None:
        checkpoint_model = (
            self.model.module if isinstance(self.model, nn.DataParallel) else self.model
        )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": checkpoint_model.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
                "metrics": {
                    "loss": metrics.loss,
                    "accuracy": metrics.accuracy,
                    "f1_score": metrics.f1_score,
                },
            },
            self.save_path / f"epoch_{epoch}.ph",
        )
