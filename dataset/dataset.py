import csv
import gzip
from os import PathLike
from pathlib import Path
from typing import Final

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


DEFAULT_SEQUENCE_LENGTH: Final = 7000
_BASE_TO_CHANNEL: Final = {"A": 0, "C": 1, "G": 2, "T": 3}
_SUPPORTED_BASES: Final = frozenset((*_BASE_TO_CHANNEL, "N"))
_ASCII_TO_CHANNEL: Final = np.full(256, 4, dtype=np.uint8)
for _base, _channel in _BASE_TO_CHANNEL.items():
    _ASCII_TO_CHANNEL[ord(_base)] = _channel


class MEIDataset(Dataset[tuple[Tensor, Tensor]]):
    """Load labeled DNA sequences as tensors accepted by VariantSeqClassify."""

    def __init__(
        self,
        csv_path: str | PathLike[str],
        sequence_length: int = DEFAULT_SEQUENCE_LENGTH,
    ) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero")

        self.sequence_length = sequence_length
        self._records = self._load_records(Path(csv_path))

    @staticmethod
    def _load_records(csv_path: Path) -> list[tuple[str, int]]:
        records: list[tuple[str, int]] = []
        # plain CSV or gzipped CSV (the shipped train set exceeds GitHub's 100 MB
        # single-file limit, so the repository stores both sets as .csv.gz)
        opener = gzip.open if csv_path.suffix == ".gz" else Path.open
        with opener(csv_path, "rt", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None or not {"sequence", "label"}.issubset(
                reader.fieldnames
            ):
                raise ValueError("CSV must contain sequence and label columns")

            for row_number, row in enumerate(reader, start=2):
                sequence = row["sequence"].strip().upper()
                invalid_bases = set(sequence) - _SUPPORTED_BASES
                if invalid_bases:
                    bases = ", ".join(sorted(invalid_bases))
                    raise ValueError(
                        f"unsupported base(s) at CSV row {row_number}: {bases}"
                    )

                try:
                    label = int(row["label"])
                except ValueError as error:
                    raise ValueError(
                        f"label at CSV row {row_number} must be an integer"
                    ) from error
                if not 0 <= label < 4:
                    raise ValueError(
                        f"label at CSV row {row_number} must be between 0 and 3"
                    )
                records.append((sequence, label))

        return records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        sequence, label = self._records[index]
        sequence = sequence[: self.sequence_length]
        encoded = torch.zeros((4, self.sequence_length), dtype=torch.float32)

        bases = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
        channels = _ASCII_TO_CHANNEL[bases]
        positions = np.flatnonzero(channels < 4)
        if positions.size:
            encoded[
                torch.from_numpy(channels[positions].astype(np.int64)), positions
            ] = 1.0

        encoded_label = torch.zeros(4, dtype=torch.float32)
        encoded_label[label] = 1.0
        return encoded, encoded_label
