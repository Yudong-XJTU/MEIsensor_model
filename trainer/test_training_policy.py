from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from trainer import TETrainer
from trainer.metrics import format_classification_table
from dataset import MEIDataset


def write_dataset_csv(path: Path) -> None:
    path.write_text("sequence,label\nACGTN,1\n", encoding="utf-8")


class TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, sequences: Tensor) -> Tensor:
        return torch.softmax(self.linear(sequences.mean(dim=2)), dim=1)


class FailingTestLoader:
    def __len__(self) -> int:
        return 1

    def __iter__(self):
        raise AssertionError("test data must not be read during train()")


def make_train_loader() -> DataLoader:
    sequences = torch.eye(4).unsqueeze(2).repeat(2, 1, 5)
    labels = torch.eye(4).repeat(2, 1)
    return DataLoader(TensorDataset(sequences, labels), batch_size=4)


def test_train_does_not_read_test_data(
    tmp_path: Path,
    capsys,
) -> None:
    trainer = TETrainer(
        TinyClassifier(),
        make_train_loader(),
        test_dataloader=FailingTestLoader(),
        with_cuda=False,
        lr=1e-6,
        save_path=tmp_path,
    )

    history = trainer.train(num_epochs=2)
    output = capsys.readouterr().out

    assert len(history) == 2
    assert "test_loss" not in output
    assert isinstance(trainer.optim, Adam)
    assert trainer.optim.param_groups[0]["lr"] == 1e-6
    assert trainer.optim.param_groups[0]["weight_decay"] == 0.0
    assert (tmp_path / "epoch_2.ph").is_file()
    assert not (tmp_path / "epoch_2.pt").exists()


def test_classification_table_uses_paper_label_order() -> None:
    report = format_classification_table(
        labels=[0, 1, 2, 3],
        predictions=[0, 1, 2, 3],
    )

    for class_name in ("Alu", "LINE1", "SVA", "Non-MEI"):
        assert class_name in report
    assert "|   0   | Non-MEI |" in report
    assert "|   1   |   Alu   |" in report
    assert "|   2   |  LINE1  |" in report
    assert "|   3   |   SVA   |" in report


def test_sequence_channels_follow_acgt_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "sequence.csv"
    write_dataset_csv(csv_path)

    sequence, label = MEIDataset(csv_path, sequence_length=5)[0]

    assert torch.equal(sequence, torch.eye(4, 5, dtype=torch.float32))
    assert torch.equal(label, torch.tensor([0.0, 1.0, 0.0, 0.0]))
