from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class ResearchAnalysis:
    """一次条件研究的统计结果。"""

    sample_size: int
    mean_return: float
    median_return: float
    win_rate: float
    max_gain: float
    max_drawdown: float
    std_return: float
    sharpe_ratio: float | None
    profit_factor: float | None
    expectancy: float


@dataclass(frozen=True)
class ResearchComparison:
    """条件组与对照组的收益比较结果。"""

    condition: ResearchAnalysis
    control: ResearchAnalysis
    mean_return_difference: float
    win_rate_difference: float


class ResearchAnalyzer:
    """对已经计算好的研究特征进行条件分析。"""

    def analyze(
        self,
        frame: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
        future_return_column: str = "future_return_5d",
    ) -> ResearchAnalysis:
        if future_return_column not in frame.columns:
            raise ValueError(
                f"missing future return column: {future_return_column}"
            )

        mask = condition(frame)

        if not isinstance(mask, pd.Series):
            raise ValueError("condition must return a pandas Series")

        mask = mask.astype(bool)

        selected = frame.loc[
            mask,
            future_return_column,
        ].dropna()

        if selected.empty:
            raise ValueError("no valid samples matched the condition")

        return self._build_analysis(selected)

    def compare(
        self,
        frame: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
        future_return_column: str = "future_return_5d",
    ) -> ResearchComparison:
        """比较条件组与对照组的未来收益。"""

        if future_return_column not in frame.columns:
            raise ValueError(
                f"missing future return column: {future_return_column}"
            )

        mask = condition(frame)

        if not isinstance(mask, pd.Series):
            raise ValueError("condition must return a pandas Series")

        mask = mask.astype(bool)

        condition_frame = frame.loc[mask]
        control_frame = frame.loc[~mask]

        condition_result = self.analyze(
            condition_frame,
            lambda data: pd.Series(True, index=data.index),
            future_return_column=future_return_column,
        )

        control_result = self.analyze(
            control_frame,
            lambda data: pd.Series(True, index=data.index),
            future_return_column=future_return_column,
        )

        return ResearchComparison(
            condition=condition_result,
            control=control_result,
            mean_return_difference=(
                condition_result.mean_return
                - control_result.mean_return
            ),
            win_rate_difference=(
                condition_result.win_rate
                - control_result.win_rate
            ),
        )

    @staticmethod
    def _build_analysis(
        selected: pd.Series,
    ) -> ResearchAnalysis:
        """从有效收益序列计算完整研究统计量。"""

        sample_size = len(selected)

        mean_return = float(selected.mean())
        median_return = float(selected.median())
        win_rate = float((selected > 0).mean())
        max_gain = float(selected.max())
        std_return = float(selected.std(ddof=1))

        if pd.isna(std_return) or abs(std_return) < 1e-12:
            sharpe_ratio = None
        else:
            sharpe_ratio = mean_return / std_return

        gains = selected[selected > 0].sum()
        losses = selected[selected < 0].sum()

        if losses == 0:
            profit_factor = None if gains == 0 else float("inf")
        else:
            profit_factor = float(gains / abs(losses))

        expectancy = mean_return

        cumulative = (1 + selected).cumprod()
        running_peak = cumulative.cummax()
        drawdown = cumulative / running_peak - 1

        max_drawdown = float(drawdown.min())

        return ResearchAnalysis(
            sample_size=sample_size,
            mean_return=mean_return,
            median_return=median_return,
            win_rate=win_rate,
            max_gain=max_gain,
            max_drawdown=max_drawdown,
            std_return=std_return,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            expectancy=expectancy,
        )
