from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research.analyzer import ResearchAnalysis, ResearchAnalyzer
from research.split import ResearchDataSplitter


@dataclass(frozen=True)
class ResearchValidationReport:
    """一次研究假设的样本内、验证集和样本外验证结果。"""

    in_sample: ResearchAnalysis
    validation: ResearchAnalysis
    out_of_sample: ResearchAnalysis

    mean_return_decay: float
    win_rate_decay: float
    stable: bool


class ResearchValidator:
    """验证研究条件在不同时间区间中的稳定性。"""

    def __init__(
        self,
        analyzer: ResearchAnalyzer | None = None,
        splitter: ResearchDataSplitter | None = None,
    ) -> None:
        self.analyzer = analyzer or ResearchAnalyzer()
        self.splitter = splitter or ResearchDataSplitter()

    def validate(
        self,
        frame: pd.DataFrame,
        condition,
        train_end: str,
        validation_end: str,
        future_return_column: str = "future_return_5d",
    ) -> ResearchValidationReport:
        split = self.splitter.split(
            frame,
            train_end=train_end,
            validation_end=validation_end,
        )

        in_sample = self.analyzer.analyze(
            split.in_sample,
            condition,
            future_return_column=future_return_column,
        )

        validation = self.analyzer.analyze(
            split.validation,
            condition,
            future_return_column=future_return_column,
        )

        out_of_sample = self.analyzer.analyze(
            split.out_of_sample,
            condition,
            future_return_column=future_return_column,
        )

        mean_return_decay = (
            out_of_sample.mean_return
            - in_sample.mean_return
        )

        win_rate_decay = (
            out_of_sample.win_rate
            - in_sample.win_rate
        )

        stable = (
            in_sample.mean_return > 0
            and validation.mean_return > 0
            and out_of_sample.mean_return > 0
        )

        return ResearchValidationReport(
            in_sample=in_sample,
            validation=validation,
            out_of_sample=out_of_sample,
            mean_return_decay=mean_return_decay,
            win_rate_decay=win_rate_decay,
            stable=stable,
        )
