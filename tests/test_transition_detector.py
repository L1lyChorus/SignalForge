import pandas as pd
import pytest

from research.transition.detector import (
    MarketStructureTransitionDetector,
)


def test_detects_state_transition():
    frame = pd.DataFrame(
        {
            "market_structure_state": [
                "VOLATILITY_COMPRESSION",
                "SIDEWAYS_ACCUMULATION",
                "TREND_EXPANSION_UP",
            ]
        }
    )

    result = MarketStructureTransitionDetector().detect(frame)

    assert result["structure_transition"].tolist() == [
        "NO_TRANSITION",
        "VOLATILITY_COMPRESSION->SIDEWAYS_ACCUMULATION",
        "SIDEWAYS_ACCUMULATION->TREND_EXPANSION_UP",
    ]


def test_unchanged_state_has_no_transition():
    frame = pd.DataFrame(
        {
            "market_structure_state": [
                "SIDEWAYS_ACCUMULATION",
                "SIDEWAYS_ACCUMULATION",
                "SIDEWAYS_ACCUMULATION",
            ]
        }
    )

    result = MarketStructureTransitionDetector().detect(frame)

    assert result["structure_transition"].tolist() == [
        "NO_TRANSITION",
        "NO_TRANSITION",
        "NO_TRANSITION",
    ]


def test_detect_latest_returns_transition():
    frame = pd.DataFrame(
        {
            "market_structure_state": [
                "VOLATILITY_COMPRESSION",
                "TREND_EXPANSION_UP",
            ]
        }
    )

    transition = (
        MarketStructureTransitionDetector()
        .detect_latest(frame)
    )

    assert transition is not None
    assert transition.previous_state == "VOLATILITY_COMPRESSION"
    assert transition.current_state == "TREND_EXPANSION_UP"
    assert (
        transition.label
        == "VOLATILITY_COMPRESSION->TREND_EXPANSION_UP"
    )


def test_detect_latest_returns_none_without_transition():
    frame = pd.DataFrame(
        {
            "market_structure_state": [
                "SIDEWAYS_ACCUMULATION",
                "SIDEWAYS_ACCUMULATION",
            ]
        }
    )

    transition = (
        MarketStructureTransitionDetector()
        .detect_latest(frame)
    )

    assert transition is None


def test_missing_state_column_raises():
    frame = pd.DataFrame(
        {
            "close": [10, 11, 12],
        }
    )

    with pytest.raises(ValueError, match="market_structure_state"):
        MarketStructureTransitionDetector().detect(frame)
