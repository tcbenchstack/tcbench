from __future__ import annotations

import polars as pl
import numpy as np

from numpy.typing import NDArray
from typing import Iterable

def confusion_matrix(
    y_true: NDArray, 
    y_pred: NDArray, 
    expected_labels: Iterable[str] = None, 
    order: str | None = "lexicographic",
    descending: bool = False,
    normalize: bool = False,
) -> pl.DataFrame:
    # compute base confusion matrix
    conf_mtx = (
        pl.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
        })	
        .group_by("y_true", "y_pred")
        .len()
        .pivot(
            index="y_true",
            on="y_pred",
            values="len"
        )
    )

    pred_labels = conf_mtx.drop("y_true").columns
    if expected_labels is None:
        expected_labels = pred_labels
    
    if order == "lexicographic":
        expected_labels = sorted(expected_labels, reverse=descending)
    elif order == "samples":
        true_labels_order = (
            pl.DataFrame({
                "y_true": conf_mtx["y_true"],
                "samples": conf_mtx.drop("y_true").sum_horizontal()
            })
            .sort("samples", descending=descending)
            .to_series()
            .to_list()
        )

        # first labels in common between true/pred
        common = set(pred_labels).intersection(set(true_labels_order))
        l = [
            col
            for col in true_labels_order
            if col in common
        ]
        # ...then the remainder
        for col in true_labels_order:
            if col in l:
                continue
            l.append(col)
        expected_labels = l

    expected_label_idx = dict(zip(
        expected_labels, range(len(expected_labels))
    ))

    # inject empty prediction columns (if needed)
    conf_mtx = conf_mtx.with_columns(
        **{
            col: None
            for col in expected_labels
            if col not in pred_labels
        }
    ).select("y_true", *expected_labels)

    conf_mtx = (
        # inject a dummy column to define rows order
        conf_mtx.with_columns(
            y_true_order=(
                pl.col("y_true")
                .map_elements(
                    function=lambda text: expected_label_idx.get(text, -1),
                    return_dtype=pl.UInt32
                )
            )
        )
        # ...and impose row order
        .sort("y_true_order", "y_true", descending=False)
        .drop("y_true_order")
        # total by row
        .with_columns(
            _total_=pl.sum_horizontal(expected_labels)
        )        
        .fill_null(0)
    )

    # total by column
    conf_mtx = pl.concat(
        (
            conf_mtx, 
            pl.concat(
                (
                    pl.DataFrame({"y_true": ["_total_"]}),
                    conf_mtx.drop("y_true").sum()
                ), 
                how="horizontal"
            )
        ),
        how="vertical"
    )

    if normalize:
        conf_mtx = (
            # normalize by row
            conf_mtx.select(
                "y_true", 
                *[
                    pl.col(col).truediv(pl.col("_total_"))
                    for col in conf_mtx.drop("y_true", "_total_").columns
                ]
            )
            # remove columns totals
            .filter(pl.col("y_true") != "_total_")
        )

    return conf_mtx


def classification_report_from_confusion_matrix(
    conf_mtx: pl.DataFrame,
    order: str = "lexicographic",
    descending: bool = False,
) -> pl.DataFrame:
    labels = conf_mtx.drop("y_true", "_total_").columns

    diag_counts = (
        conf_mtx
            .drop("y_true", "_total_")
            .to_numpy()
            [np.diag_indices(len(labels))]
    )
    true_counts = (
        conf_mtx
            .filter(pl.col("y_true") != "_total_")
            ["_total_"]
            .to_numpy()
    )
    pred_counts = (
        conf_mtx
        .filter(pl.col("y_true") == "_total_")
        .drop("y_true", "_total_")
        .to_numpy()
        .squeeze()
    )

    recall = diag_counts / true_counts
    precision = diag_counts / pred_counts
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = diag_counts.sum() / true_counts.sum()

    schema={
        "label": pl.String,
        "precision": pl.Float32,
        "recall": pl.Float32,
        "f1-score": pl.Float32,
        "support": pl.Float32
    }

    class_rep = (
        pl.DataFrame(
            {
                "label": labels,
                "precision": precision,
                "recall": recall,
                "f1-score": f1_score,
                "support": true_counts,
            }, 
            schema=schema
        )
    )
    if order == "lexicographic":
        class_rep = class_rep.sort("label", descending=descending)
    elif order == "samples":
        class_rep = class_rep.sort("support", descending=descending)

    extra_metrics = (
        pl.DataFrame(
            {
                "label": [
                    "accuracy",
                    "macro avg",
                    "weighted avg",
                ],
                "precision": [
                    accuracy,
                    precision.mean(),
                    np.dot(precision, true_counts) / true_counts.sum(),
                ],
                "recall": [
                    accuracy,  
                    recall.mean(),
                    np.dot(recall, true_counts) / true_counts.sum(),
                ],
                "f1-score": [
                    accuracy,
                    f1_score.mean(),
                    np.dot(f1_score, true_counts) / true_counts.sum(),
                ],
                "support": [
                    accuracy,
                    true_counts.sum(),
                    true_counts.sum(),
                ]
            }, 
            schema=schema
        )
    )

    return pl.concat((class_rep, extra_metrics))

def classification_report(
    y_true: NDArray,
    y_pred: NDArray,
    expected_labels: Iterable[str] = None,
    order: str = "lexicographic",
    descending: bool = False,
) -> pl.DataFrame:
    conf_mtx = confusion_matrix(
        y_true, 
        y_pred, 
        expected_labels,
        order=order,
        descending=descending,
        normalize=False,
    )
    return classification_report_from_confusion_matrix(
        conf_mtx, 
        order=order, 
        descending=descending
    )
