from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import pandas as pd

from research.analyzer import ResearchAnalysis, ResearchAnalyzer


@dataclass(frozen=True)
class RobustnessCase:
    """一次参数变化下的稳健性测试结果。"""

    name: str
    analysis: ResearchAnalysis
    sample_size_valid: bool
    return_valid: bool
    sharpe_valid: bool

    @property
    def is_valid(self) -> bool:
        return (
            self.sample_size_valid
            and self.return_valid
            and self.sharpe_valid
        )


@dataclass(frozen=True)
class RobustnessAnalysis:
    """同一研究条件在多个参数下的稳健性结果。"""

    cases: List[RobustnessCase]
    valid_case_count: int
    total_case_count: int

    @property
    def stability_ratio(self) -> float:
        if self.total_case_count == 0:
            return 0.0

        return self.valid_case_count / self.total_case_count


class ResearchRobustnessAnalyzer:
    """测试研究条件在不同参数下是否保持有效。"""

    def __init__(
        self,
        analyzer: ResearchAnalyzer | None = None,
    ) -> None:
        self.analyzer = analyzer or ResearchAnalyzer()

    def analyze(
        self,
        frame: pd.DataFrame,
        conditions: Dict[str, Callable[[pd.DataFrame], pd.Series]],
        future_return_column: str = "future_return_5d",
        min_sample_size: int = 30,
    ) -> RobustnessAnalysis:
        if not conditions:
            raise ValueError("conditions must not be empty")

        if min_sample_size <= 0:
            raise ValueError("min_sample_size must be greater than 0")

        cases: List[RobustnessCase] = []

        for name, condition in conditions.items():
            analysis = self.analyzer.analyze(
                frame,
                condition,
                future_return_column=future_return_column,
            )

            sample_size_valid = (
                analysis.sample_size >= min_sample_size
            )

            return_valid = analysis.mean_return > 0

            sharpe_valid = (
                analysis.sharpe_ratio is not None
                and analysis.sharpe_ratio > 0
            )

            cases.append(
                RobustnessCase(
                    name=name,
                    analysis=analysis,
                    sample_size_valid=sample_size_valid,
                    return_valid=return_valid,
                    sharpe_valid=sharpe_valid,
                )
            )

        valid_case_count = sum(
            1
            for case in cases
            if case.is_valid
        )

        return RobustnessAnalysis(
            cases=cases,
            valid_case_count=valid_case_count,
            total_case_count=len(cases),
        )
