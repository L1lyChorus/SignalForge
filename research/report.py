from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from research.analyzer import ResearchAnalysis
from research.robustness import RobustnessAnalysis


@dataclass(frozen=True)
class ResearchReport:
    """一次量化研究的统一结果报告。"""

    hypothesis_id: str
    hypothesis_title: str

    in_sample: Optional[ResearchAnalysis] = None
    validation: Optional[ResearchAnalysis] = None
    out_of_sample: Optional[ResearchAnalysis] = None

    robustness: Optional[RobustnessAnalysis] = None

    purged_sample_size: int = 0

    conclusion: str = ""

    def validate(self) -> None:
        if not self.hypothesis_id.strip():
            raise ValueError("hypothesis_id is required")

        if not self.hypothesis_title.strip():
            raise ValueError("hypothesis_title is required")

        if (
            self.in_sample is None
            and self.validation is None
            and self.out_of_sample is None
        ):
            raise ValueError(
                "at least one research analysis is required"
            )

        if self.purged_sample_size < 0:
            raise ValueError(
                "purged_sample_size must be greater than or equal to 0"
            )

    @property
    def stability_ratio(self) -> float | None:
        if self.robustness is None:
            return None

        return self.robustness.stability_ratio

    @property
    def out_of_sample_mean_return(self) -> float | None:
        if self.out_of_sample is None:
            return None

        return self.out_of_sample.mean_return

    @property
    def out_of_sample_sharpe(self) -> float | None:
        if self.out_of_sample is None:
            return None

        return self.out_of_sample.sharpe_ratio
