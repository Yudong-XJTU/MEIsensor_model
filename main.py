import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from dataset import MEIDataset
from model.model import VariantSeqClassify
from trainer import TETrainer


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    train_csv: Path
    test_csv: Path
    save_path: Path
    epochs: int
    batch_size: int
    learning_rate: float
    num_workers: int
    log_freq: int
    seed: int
    smoke: bool
    disable_cudnn: bool


def parse_args() -> TrainingConfig:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train the MEI sequence classifier")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=project_root / "dataset" / "data" / "MEI_four_class_train.csv.gz",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=project_root / "dataset" / "data" / "MEI_four_class_test.csv.gz",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=project_root / "fine_tuned_model",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-freq", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one epoch with 64 train and 32 test samples",
    )
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="Use CUDA without the cuDNN backend",
    )
    args = parser.parse_args()
    return TrainingConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        log_freq=args.log_freq,
        seed=args.seed,
        smoke=args.smoke,
        disable_cudnn=args.disable_cudnn,
    )


def limit_dataset(dataset: Dataset, sample_count: int) -> Dataset:
    return Subset(dataset, range(min(sample_count, len(dataset))))


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    config = parse_args()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.enabled = not config.disable_cudnn
    torch.backends.cudnn.benchmark = False

    train_dataset: Dataset = MEIDataset(config.train_csv)
    test_dataset: Dataset = MEIDataset(config.test_csv)
    epochs = config.epochs
    if config.smoke:
        train_dataset = limit_dataset(train_dataset, 64)
        test_dataset = limit_dataset(test_dataset, 32)
        epochs = 1

    train_dataloader = build_dataloader(
        train_dataset,
        config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    test_dataloader = build_dataloader(
        test_dataset,
        config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    print("Training configuration:")
    print("  train samples:", len(train_dataset))
    print("  test samples:", len(test_dataset))
    print("  epochs:", epochs)
    print("  batch size:", config.batch_size)
    print("  learning rate:", config.learning_rate)
    print("  optimizer: Adam")
    print("  loss: categorical cross-entropy")
    print("  CUDA available:", torch.cuda.is_available())
    print("  cuDNN enabled:", torch.backends.cudnn.enabled)
    print("  checkpoint path:", config.save_path)

    trainer = TETrainer(
        model=VariantSeqClassify(),
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        with_cuda=True,
        lr=config.learning_rate,
        cuda_devices=None,
        log_freq=config.log_freq,
        save_path=config.save_path,
    )
    trainer.train(num_epochs=epochs)
    trainer.test()


if __name__ == "__main__":
    main()
