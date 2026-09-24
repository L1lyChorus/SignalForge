from __future__ import annotations

from enum import Enum

import pandas as pd


class VolumePriceState(str, Enum):
    """量价结构状态。"""

    EXPANSION_UP = "EXPANSION_UP"
    EXPANSION_DOWN = "EXPANSION_DOWN"
    ACCUMULATION = "ACCUMULATION"
    DISTRIBUTION = "DISTRIBUTION"
    CONTRACTION = "CONTRACTION"
    NEUTRAL = "NEUTRAL"
    UNKNOWN = "UNKNOWN"


class VolumePriceStructureDetector:
    """
    根据成交量、价格变化和收盘位置识别量价结构。

    注意：
    这里只负责描述市场状态，不产生交易信号。
    """

    def __init__(
        self,
        volume_window: int = 20,
        expansion_threshold: float = 2.0,
        contraction_threshold: float = 0.7,
        price_threshold: float = 0.01,
        close_location_threshold: float = 0.7,
    ) -> None:
        if volume_window <= 0:
            raise ValueError(
                "volume_window must be greater than 0"
            )

        if expansion_threshold <= 1.0:
            raise ValueError(
                "expansion_threshold must be greater than 1"
            )

        if not 0 < contraction_threshold < 1:
            raise ValueError(
                "contraction_threshold must be between 0 and 1"
            )

        if price_threshold < 0:
            raise ValueError(
                "price_threshold must not be negative"
            )

        if not 0.5 <= close_location_threshold <= 1:
            raise ValueError(
                "close_location_threshold must be between 0.5 and 1"
            )

        self.volume_window = volume_window
        self.expansion_threshold = expansion_threshold
        self.contraction_threshold = contraction_threshold
        self.price_threshold = price_threshold
        self.close_location_threshold = close_location_threshold

    def detect(self, frame: pd.DataFrame) -> pd.Series:
        required = {
            "high",
            "low",
            "close",
            "volume",
        }

        missing = required - set(frame.columns)

        if missing:
            raise ValueError(
                f"missing required columns: {sorted(missing)}"
            )

        high = pd.to_numeric(frame["high"], errors="coerce")
        low = pd.to_numeric(frame["low"], errors="coerce")
        close = pd.to_numeric(frame["close"], errors="coerce")
        volume = pd.to_numeric(frame["volume"], errors="coerce")

        # 只使用当前K线之前的成交量计算基准，
        # 避免把当天的成交量混入自己的比较基准。
        average_volume = (
            volume
            .rolling(self.volume_window)
            .mean()
            .shift(1)
        )

        relative_volume = volume / average_volume

        price_return = close.pct_change()

        price_range = high - low

        close_location = (
            (close - low)
            / price_range.replace(0, pd.NA)
        )

        result = pd.Series(
            VolumePriceState.UNKNOWN.value,
            index=frame.index,
            dtype="object",
        )

        valid = (
            relative_volume.notna()
            & price_return.notna()
            & close_location.notna()
        )

        expansion = (
            valid
            & (relative_volume >= self.expansion_threshold)
        )

        contraction = (
            valid
            & (relative_volume <= self.contraction_threshold)
        )

        strong_close = (
            close_location
            >= self.close_location_threshold
        )

        weak_close = (
            close_location
            <= (1 - self.close_location_threshold)
        )

        expansion_up = (
            expansion
            & (price_return >= self.price_threshold)
            & strong_close
        )

        expansion_down = (
            expansion
            & (price_return <= -self.price_threshold)
            & weak_close
        )

        accumulation = (
            expansion
            & (price_return > -self.price_threshold)
            & strong_close
            & ~expansion_up
        )

        distribution = (
            expansion
            & (price_return < self.price_threshold)
            & weak_close
            & ~expansion_down
        )

        result.loc[contraction] = (
            VolumePriceState.CONTRACTION.value
        )

        result.loc[expansion_up] = (
            VolumePriceState.EXPANSION_UP.value
        )

        result.loc[expansion_down] = (
            VolumePriceState.EXPANSION_DOWN.value
        )

        result.loc[accumulation] = (
            VolumePriceState.ACCUMULATION.value
        )

        result.loc[distribution] = (
            VolumePriceState.DISTRIBUTION.value
        )

        neutral = (
            valid
            & result.eq(VolumePriceState.UNKNOWN.value)
        )

        result.loc[neutral] = (
            VolumePriceState.NEUTRAL.value
        )

        return result
