from __future__ import annotations

from typing import Callable

import pandas as pd

from research.analyzer import ResearchAnalysis
from research.experiment import ResearchExperiment
from research.ledger import ResearchLedger
from research.models import ResearchResult
from research.pipeline import ResearchPipeline
from research.purged_split import PurgedResearchDataSplitter
from research.report import ResearchReport
from research.robustness import RobustnessAnalysis


class ResearchExperimentRunner:
    """执行研究实验，并将结果登记到 ResearchLedger。"""

    def __init__(
        self,
        ledger: ResearchLedger,
        pipeline: ResearchPipeline | None = None,
        splitter: PurgedResearchDataSplitter | None = None,
    ) -> None:
        self.ledger = ledger
        self.pipeline = pipeline or ResearchPipeline()
        self.splitter = splitter or PurgedResearchDataSplitter()

    def run(
        self,
        experiment: ResearchExperiment,
        frame: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
        future_return_column: str = "future_return_5d",
    ) -> ResearchResult:
        """执行一次完整研究实验并登记结果。"""

        experiment.validate()

        registered_experiment = self.ledger.get_experiment(
            experiment.experiment_id
        )

        if registered_experiment.hypothesis_id != experiment.hypothesis_id:
            raise ValueError(
                "experiment does not belong to hypothesis"
            )

        self.ledger.update_experiment_status(
            experiment.experiment_id,
            "RUNNING",
        )

        featured = self.pipeline.prepare_features(frame)

        analysis = self.pipeline.analyzer.analyze(
            featured,
            condition,
            future_return_column=future_return_column,
        )

        result = ResearchResult(
            hypothesis_id=experiment.hypothesis_id,
            sample_size=analysis.sample_size,
            mean_return=analysis.mean_return,
            median_return=analysis.median_return,
            win_rate=analysis.win_rate,
            max_gain=analysis.max_gain,
            max_drawdown=analysis.max_drawdown,
            experiment_id=experiment.experiment_id,
            sharpe_ratio=analysis.sharpe_ratio,
        )

        self.ledger.record_result(result)

        self.ledger.update_experiment_status(
            experiment.experiment_id,
            "COMPLETED",
        )

        return result

    def run_validation(
        self,
        experiment: ResearchExperiment,
        frame: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
        train_end: str,
        validation_end: str,
        purge_days: int = 0,
        future_return_column: str = "future_return_5d",
        hypothesis_title: str = "",
    ) -> ResearchReport:
        """
        执行带 purge 的三阶段研究验证。

        数据流程：

        原始数据
            ↓
        特征计算
            ↓
        时间切分 + purge
            ↓
        In-Sample / Validation / Out-of-Sample
            ↓
        ResearchReport
        """

        experiment.validate()

        registered_experiment = self.ledger.get_experiment(
            experiment.experiment_id
        )

        if registered_experiment.hypothesis_id != experiment.hypothesis_id:
            raise ValueError(
                "experiment does not belong to hypothesis"
            )

        if not hypothesis_title.strip():
            hypothesis = self.ledger.get_hypothesis(
                experiment.hypothesis_id
            )
            hypothesis_title = hypothesis.title

        featured = self.pipeline.prepare_features(frame)

        split = self.splitter.split(
            featured,
            train_end=train_end,
            validation_end=validation_end,
            purge_days=purge_days,
        )

        in_sample = self.pipeline.analyzer.analyze(
            split.in_sample,
            condition,
            future_return_column=future_return_column,
        )

        validation = self.pipeline.analyzer.analyze(
            split.validation,
            condition,
            future_return_column=future_return_column,
        )

        out_of_sample = self.pipeline.analyzer.analyze(
            split.out_of_sample,
            condition,
            future_return_column=future_return_column,
        )

        report = ResearchReport(
            hypothesis_id=experiment.hypothesis_id,
            hypothesis_title=hypothesis_title,
            in_sample=in_sample,
            validation=validation,
            out_of_sample=out_of_sample,
            purged_sample_size=len(split.purged),
        )

        report.validate()

        return report

    @staticmethod
    def build_report(
        experiment: ResearchExperiment,
        hypothesis_title: str,
        in_sample: ResearchAnalysis | None = None,
        validation: ResearchAnalysis | None = None,
        out_of_sample: ResearchAnalysis | None = None,
        robustness: RobustnessAnalysis | None = None,
        purged_sample_size: int = 0,
        conclusion: str = "",
    ) -> ResearchReport:
        """将研究阶段结果统一封装成 ResearchReport。"""

        report = ResearchReport(
            hypothesis_id=experiment.hypothesis_id,
            hypothesis_title=hypothesis_title,
            in_sample=in_sample,
            validation=validation,
            out_of_sample=out_of_sample,
            robustness=robustness,
            purged_sample_size=purged_sample_size,
            conclusion=conclusion,
        )

        report.validate()

        return report
