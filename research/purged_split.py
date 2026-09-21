from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class PurgedResearchDataSplit:
    """按时间切分并移除边界污染样本后的研究数据集。"""

    in_sample: pd.DataFrame
    validation: pd.DataFrame
    out_of_sample: pd.DataFrame
    purged: pd.DataFrame


class PurgedResearchDataSplitter:
    """
    按时间切分研究数据，并在训练集与验证集之间
    留出 purge window，降低未来收益标签造成的时间泄漏。
    """

    def split(
        self,
        frame: pd.DataFrame,
        train_end: str,
        validation_end: str,
        purge_days: int = 0,
    ) -> PurgedResearchDataSplit:
        if "datetime" not in frame.columns:
            raise ValueError("missing required column: datetime")

        if frame.empty:
            raise ValueError("frame must not be empty")

        if purge_days < 0:
            raise ValueError("purge_days must be greater than or equal to 0")

        train_end_ts = pd.to_datetime(
            train_end,
            utc=True,
            errors="coerce",
        )
        validation_end_ts = pd.to_datetime(
            validation_end,
            utc=True,
            errors="coerce",
        )

        if pd.isna(train_end_ts):
            raise ValueError("train_end must be a valid datetime")

        if pd.isna(validation_end_ts):
            raise ValueError("validation_end must be a valid datetime")

        if train_end_ts >= validation_end_ts:
            raise ValueError(
                "train_end must be before validation_end"
            )

        result = frame.copy()

        result["datetime"] = pd.to_datetime(
            result["datetime"],
            utc=True,
            errors="coerce",
        )

        if result["datetime"].isna().any():
            raise ValueError("datetime contains invalid values")

        result = (
            result
            .sort_values("datetime", kind="stable")
            .reset_index(drop=True)
        )

        purge_end_ts = train_end_ts + pd.Timedelta(
            days=purge_days
        )

        in_sample = result[
            result["datetime"] <= train_end_ts
        ].copy()

        purged = result[
            (result["datetime"] > train_end_ts)
            & (result["datetime"] <= purge_end_ts)
            & (result["datetime"] <= validation_end_ts)
        ].copy()

        validation = result[
            (result["datetime"] > purge_end_ts)
            & (result["datetime"] <= validation_end_ts)
        ].copy()

        out_of_sample = result[
            result["datetime"] > validation_end_ts
        ].copy()

        return PurgedResearchDataSplit(
            in_sample=in_sample.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            out_of_sample=out_of_sample.reset_index(drop=True),
            purged=purged.reset_index(drop=True),
        )
