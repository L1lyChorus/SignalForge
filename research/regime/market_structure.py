from __future__ import annotations

from enum import Enum

import pandas as pd

from research.regime.volume_price import VolumePriceState


class MarketStructureState(str, Enum):
    """
    综合市场结构状态。

    这里只描述市场行为，不产生交易信号。
    """

    TREND_ACCUMULATION = "TREND_ACCUMULATION"
    TREND_EXPANSION_UP = "TREND_EXPANSION_UP"
    TREND_DISTRIBUTION = "TREND_DISTRIBUTION"
    TREND_EXPANSION_DOWN = "TREND_EXPANSION_DOWN"

    SIDEWAYS_ACCUMULATION = "SIDEWAYS_ACCUMULATION"
    SIDEWAYS_DISTRIBUTION = "SIDEWAYS_DISTRIBUTION"
    VOLATILITY_COMPRESSION = "VOLATILITY_COMPRESSION"

    UNKNOWN = "UNKNOWN"


class MarketStructureDetector:
    """
    将市场 Regime 与量价结构组合成更高层的市场结构。

    注意：
    1. 不产生 BUY / SELL。
    2. 不进行未来收益计算。
    3. 不包含策略参数。
    4. 只描述当前可观察到的市场状态。
    """

    def detect(self, frame: pd.DataFrame) -> pd.Series:
        required = {
            "trend_state",
            "volatility_state",
            "liquidity_state",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"missing required columns: {sorted(missing)}"
            )

        if "volume_price_state" not in frame.columns:
            raise ValueError(
                "missing required column: volume_price_state"
            )

        result = pd.Series(
            MarketStructureState.UNKNOWN.value,
            index=frame.index,
            dtype="object",
        )

        trend = frame["trend_state"]
        volatility = frame["volatility_state"]
        volume_price = frame["volume_price_state"]

        expansion_up = volume_price.eq(
            VolumePriceState.EXPANSION_UP.value
        )

        expansion_down = volume_price.eq(
            VolumePriceState.EXPANSION_DOWN.value
        )

        accumulation = volume_price.eq(
            VolumePriceState.ACCUMULATION.value
        )

        distribution = volume_price.eq(
            VolumePriceState.DISTRIBUTION.value
        )

        contraction = volume_price.eq(
            VolumePriceState.CONTRACTION.value
        )

        bull = trend.eq("BULL")
        bear = trend.eq("BEAR")
        sideways = trend.eq("SIDEWAYS")

        high_volatility = volatility.eq("HIGH")
        low_volatility = volatility.eq("LOW")

        result.loc[
            bull & accumulation
        ] = MarketStructureState.TREND_ACCUMULATION.value

        result.loc[
            bull & expansion_up
        ] = MarketStructureState.TREND_EXPANSION_UP.value

        result.loc[
            bull & distribution
        ] = MarketStructureState.TREND_DISTRIBUTION.value

        result.loc[
            bear & expansion_down
        ] = MarketStructureState.TREND_EXPANSION_DOWN.value

        result.loc[
            sideways & accumulation
        ] = MarketStructureState.SIDEWAYS_ACCUMULATION.value

        result.loc[
            sideways & distribution
        ] = MarketStructureState.SIDEWAYS_DISTRIBUTION.value

        result.loc[
            low_volatility & contraction
        ] = MarketStructureState.VOLATILITY_COMPRESSION.value

        # 高波动但没有明确结构时，不强行解释。
        # 保留 UNKNOWN / 其他已经识别出的状态。

        return result
