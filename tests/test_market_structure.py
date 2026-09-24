import pandas as pd
import pytest

from research.regime.market_structure import (
    MarketStructureDetector,
    MarketStructureState,
)


def make_frame(
    trend,
    volatility,
    liquidity,
    volume_price,
):
    return pd.DataFrame(
        {
            "trend_state": trend,
            "volatility_state": volatility,
            "liquidity_state": liquidity,
            "volume_price_state": volume_price,
        }
    )


def test_bull_accumulation_is_trend_accumulation():
    frame = make_frame(
        ["BULL"],
        ["NORMAL"],
        ["NORMAL"],
        ["ACCUMULATION"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.TREND_ACCUMULATION.value
    )


def test_bull_expansion_up_is_trend_expansion_up():
    frame = make_frame(
        ["BULL"],
        ["NORMAL"],
        ["HIGH"],
        ["EXPANSION_UP"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.TREND_EXPANSION_UP.value
    )


def test_bull_distribution_is_trend_distribution():
    frame = make_frame(
        ["BULL"],
        ["NORMAL"],
        ["HIGH"],
        ["DISTRIBUTION"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.TREND_DISTRIBUTION.value
    )


def test_bear_expansion_down_is_trend_expansion_down():
    frame = make_frame(
        ["BEAR"],
        ["HIGH"],
        ["HIGH"],
        ["EXPANSION_DOWN"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.TREND_EXPANSION_DOWN.value
    )


def test_sideways_accumulation_is_sideways_accumulation():
    frame = make_frame(
        ["SIDEWAYS"],
        ["NORMAL"],
        ["NORMAL"],
        ["ACCUMULATION"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.SIDEWAYS_ACCUMULATION.value
    )


def test_sideways_distribution_is_sideways_distribution():
    frame = make_frame(
        ["SIDEWAYS"],
        ["NORMAL"],
        ["HIGH"],
        ["DISTRIBUTION"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.SIDEWAYS_DISTRIBUTION.value
    )


def test_low_volatility_contraction_is_compression():
    frame = make_frame(
        ["SIDEWAYS"],
        ["LOW"],
        ["LOW"],
        ["CONTRACTION"],
    )

    result = MarketStructureDetector().detect(frame)

    assert result.iloc[-1] == (
        MarketStructureState.VOLATILITY_COMPRESSION.value
    )


def test_missing_regime_column_raises():
    frame = pd.DataFrame(
        {
            "trend_state": ["BULL"],
            "volume_price_state": ["ACCUMULATION"],
        }
    )

    with pytest.raises(ValueError, match="missing required column"):
        MarketStructureDetector().detect(frame)


def test_missing_volume_price_column_raises():
    frame = pd.DataFrame(
        {
            "trend_state": ["BULL"],
            "volatility_state": ["NORMAL"],
            "liquidity_state": ["NORMAL"],
        }
    )

    with pytest.raises(ValueError, match="volume_price_state"):
        MarketStructureDetector().detect(frame)
