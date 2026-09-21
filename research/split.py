from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchDataSplit:
    """按时间切分后的研究数据集。"""

    in_sample: pd.DataFrame
    validation: pd.DataFrame
    out_of_sample: pd.DataFrame


class ResearchDataSplitter:
    """将研究数据按时间顺序切分，避免未来数据进入历史研究。"""

    def split(
        self,
        frame: pd.DataFrame,
        train_end: str,
        validation_end: str,
    ) -> ResearchDataSplit:
        if "datetime" not in frame.columns:
            raise ValueError("missing required column: datetime")

        if frame.empty:
            raise ValueError("frame must not be empty")

        train_end_ts = pd.to_datetime(train_end, utc=True, errors="coerce")
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

        result = result.sort_values(
            "datetime",
            kind="stable",
        ).reset_index(drop=True)

        in_sample = result[
            result["datetime"] <= train_end_ts
        ].copy()

        validation = result[
            (result["datetime"] > train_end_ts)
            & (result["datetime"] <= validation_end_ts)
        ].copy()

        out_of_sample = result[
            result["datetime"] > validation_end_ts
        ].copy()

        return ResearchDataSplit(
            in_sample=in_sample.reset_index(drop=True),
            validation=validation.reset_index(drop=True),
            out_of_sample=out_of_sample.reset_index(drop=True),
        )
