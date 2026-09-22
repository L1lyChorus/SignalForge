from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ResearchLabel:
    """研究用未来收益标签的定义。"""

    name: str
    horizon: int

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("name is required")

        if self.horizon <= 0:
            raise ValueError(
                "horizon must be greater than 0"
            )


class FutureReturnLabelBuilder:
    """生成未来 N 个交易日收益标签。"""

    def build(
        self,
        frame: pd.DataFrame,
        horizon: int,
        price_column: str = "close",
    ) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("frame must not be empty")

        if horizon <= 0:
            raise ValueError(
                "horizon must be greater than 0"
            )

        if price_column not in frame.columns:
            raise ValueError(
                f"missing required price column: {price_column}"
            )

        result = frame.copy()

        result[f"future_return_{horizon}d"] = (
            result[price_column].shift(-horizon)
            / result[price_column]
            - 1
        )

        return result

    @staticmethod
    def valid_mask(
        frame: pd.DataFrame,
        horizon: int,
        price_column: str = "close",
    ) -> pd.Series:
        if horizon <= 0:
            raise ValueError(
                "horizon must be greater than 0"
            )

        if price_column not in frame.columns:
            raise ValueError(
                f"missing required price column: {price_column}"
            )

        return frame[price_column].shift(-horizon).notna()
