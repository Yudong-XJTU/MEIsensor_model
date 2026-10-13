from dataclasses import dataclass

from sklearn.metrics import f1_score, precision_recall_fscore_support


CLASS_NAMES = ("Non-MEI", "Alu", "LINE1", "SVA")


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    loss: float
    accuracy: float
    f1_score: float


def calculate_metrics(
    total_loss: float,
    total_correct: int,
    labels: list[int],
    predictions: list[int],
) -> EpochMetrics:
    sample_count = len(labels)
    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=total_correct * 100.0 / sample_count,
        f1_score=f1_score(
            labels,
            predictions,
            average="weighted",
            zero_division=0,
        ),
    )


def format_classification_table(
    labels: list[int],
    predictions: list[int],
) -> str:
    precision, recall, f1_values, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=[0, 1, 2, 3],
        zero_division=0,
    )
    lines = [
        "+-------+---------+-----------+--------+--------+---------+",
        "| Label | Class   | Precision | Recall | F1     | Support |",
        "+-------+---------+-----------+--------+--------+---------+",
    ]
    for class_id, class_name in enumerate(CLASS_NAMES):
        lines.append(
            f"| {class_id:^5} | {class_name:^7} | {precision[class_id]:^9.4f} | "
            f"{recall[class_id]:^6.4f} | {f1_values[class_id]:^6.4f} | "
            f"{support[class_id]:^7} |"
        )
    lines.append("+-------+---------+-----------+--------+--------+---------+")
    return "\n".join(lines)
