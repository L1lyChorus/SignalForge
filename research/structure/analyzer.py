from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class StructureAnalysis:
    """一次市场结构条件研究的统计结果。"""

    condition_name: str
    sample_size: int
    mean_return: Optional[float]
    median_return: Optional[float]
    win_rate: Optional[float]
    sharpe_ratio: Optional[float]
    max_drawdown: Optional[float]
    return_quantiles: dict[str, Optional[float]]


class StructureAnalyzer:
    """
    研究某种市场结构出现以后，未来收益的统计特征。

    注意：
    1. 这里只做研究，不产生交易信号。
    2. 条件只能使用当前时点及之前的数据。
    3. 未来收益只能来自已经构造好的 future_return_* 标签。
    """

    def __init__(
        self,
        min_sample_size: int = 5,
    ) -> None:
        if min_sample_size <= 0:
            raise ValueError(
                "min_sample_size must be greater than 0"
            )

        self.min_sample_size = min_sample_size

    def analyze(
        self,
        frame: pd.DataFrame,
        condition: pd.Series,
        future_return_columns: Iterable[str],
        condition_name: str = "",
    ) -> dict[str, StructureAnalysis]:
        """分析一个条件在多个未来收益周期下的表现。"""

        if len(condition) != len(frame):
            raise ValueError(
                "condition length must match frame length"
            )

        if not isinstance(condition, pd.Series):
            raise ValueError(
                "condition must be a pandas Series"
            )

        selected = frame.loc[condition.astype(bool)].copy()

        results: dict[str, StructureAnalysis] = {}

        for column in future_return_columns:
            if column not in selected.columns:
                raise ValueError(
                    f"missing required column: {column}"
                )

            returns = pd.to_numeric(
                selected[column],
                errors="coerce",
            ).dropna()

            sample_size = len(returns)

            if sample_size < self.min_sample_size:
                results[column] = StructureAnalysis(
                    condition_name=condition_name,
                    sample_size=sample_size,
                    mean_return=None,
                    median_return=None,
                    win_rate=None,
                    sharpe_ratio=None,
                    max_drawdown=None,
                    return_quantiles={
                        "q10": None,
                        "q25": None,
                        "q50": None,
                        "q75": None,
                        "q90": None,
                    },
                )
                continue

            mean_return = float(returns.mean())
            median_return = float(returns.median())
            win_rate = float((returns > 0).mean())

            std = float(returns.std(ddof=1))

            if std == 0:
                sharpe_ratio = None
            else:
                sharpe_ratio = float(
                    mean_return / std
                )

            equity = (1.0 + returns).cumprod()
            running_max = equity.cummax()

            drawdown = (
                equity / running_max
            ) - 1.0

            max_drawdown = float(drawdown.min())

            quantiles = returns.quantile(
                [0.10, 0.25, 0.50, 0.75, 0.90]
            )

            results[column] = StructureAnalysis(
                condition_name=condition_name,
                sample_size=sample_size,
                mean_return=mean_return,
                median_return=median_return,
                win_rate=win_rate,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                return_quantiles={
                    "q10": float(quantiles.loc[0.10]),
                    "q25": float(quantiles.loc[0.25]),
                    "q50": float(quantiles.loc[0.50]),
                    "q75": float(quantiles.loc[0.75]),
                    "q90": float(quantiles.loc[0.90]),
                },
            )

        return results
