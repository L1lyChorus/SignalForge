from __future__ import annotations

from typing import Callable

import pandas as pd

from research.analyzer import ResearchAnalysis, ResearchAnalyzer
from research.features.basic import add_basic_features
from research.features.candlestick import add_candlestick_features
from research.labels import FutureReturnLabelBuilder
from research.regime.detector import MarketRegimeDetector
from research.regime.volume_price import VolumePriceStructureDetector
from research.regime.market_structure import MarketStructureDetector
from research.structure.analyzer import StructureAnalyzer

from research.transition.detector import (
    MarketStructureTransitionDetector,
)


class ResearchPipeline:
    """统一执行市场特征、研究标签与条件研究。"""

    def __init__(
        self,
        analyzer: ResearchAnalyzer | None = None,
        regime_detector: MarketRegimeDetector | None = None,
        label_builder: FutureReturnLabelBuilder | None = None,
        volume_price_detector: VolumePriceStructureDetector | None = None,
        market_structure_detector: MarketStructureDetector | None = None,
        structure_analyzer: StructureAnalyzer | None = None,
        transition_detector: MarketStructureTransitionDetector | None = None,
    ) -> None:
        self.analyzer = analyzer or ResearchAnalyzer()
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.label_builder = label_builder or FutureReturnLabelBuilder()

        self.volume_price_detector = (
            volume_price_detector or VolumePriceStructureDetector()
        )

        self.market_structure_detector = (
            market_structure_detector or MarketStructureDetector()
        )
        self.structure_analyzer = (
            structure_analyzer or StructureAnalyzer()
        )

        self.transition_detector = (
            transition_detector or MarketStructureTransitionDetector()
        )

    def prepare_features(
        self,
        frame: pd.DataFrame,
        label_horizons: tuple[int, ...] = (1, 5),
    ) -> pd.DataFrame:
        """计算基础特征、研究标签、K线结构和市场状态。"""

        result = add_basic_features(frame)

        for horizon in label_horizons:
            result = self.label_builder.build(
                result,
                horizon=horizon,
            )

        result = add_candlestick_features(result)

        result = self.regime_detector.detect(result)

        result["volume_price_state"] = (
            self.volume_price_detector.detect(result)
        )

        result["market_structure_state"] = (
            self.market_structure_detector.detect(result)
        )

        result = self.transition_detector.detect(result)

        return result

    def analyze_structure(
        self,
        frame: pd.DataFrame,
        structure_state: str,
        future_return_columns: tuple[str, ...] = (
            "future_return_1d",
            "future_return_5d",
        ),
    ):
        """研究某一种市场结构出现后的未来收益。"""

        featured = self.prepare_features(frame)

        if "market_structure_state" not in featured.columns:
            raise ValueError(
                "missing required column: market_structure_state"
            )

        condition = (
            featured["market_structure_state"]
            == structure_state
        )

        return self.structure_analyzer.analyze(
            featured,
            condition,
            future_return_columns=future_return_columns,
            condition_name=structure_state,
        )

    def analyze(
        self,
        frame: pd.DataFrame,
        condition: Callable[[pd.DataFrame], pd.Series],
        future_return_column: str = "future_return_5d",
    ) -> ResearchAnalysis:
        """计算特征后执行条件研究。"""

        featured = self.prepare_features(frame)

        return self.analyzer.analyze(
            featured,
            condition,
            future_return_column=future_return_column,
        )
