import pandas as pd
import pytest

from research.structure.analyzer import StructureAnalyzer


def make_frame():
    return pd.DataFrame(
        {
            "market_structure_state": [
                "TREND_EXPANSION_UP",
                "TREND_EXPANSION_UP",
                "TREND_DISTRIBUTION",
                "TREND_EXPANSION_UP",
                "SIDEWAYS_ACCUMULATION",
                "TREND_EXPANSION_UP",
            ],
            "future_return_1d": [
                0.02,
                -0.01,
                0.03,
                0.04,
                -0.02,
                0.01,
            ],
            "future_return_5d": [
                0.05,
                -0.02,
                0.01,
                0.08,
                -0.01,
                0.03,
            ],
        }
    )


def test_structure_analyzer_calculates_statistics():

    frame = make_frame()

    condition = (
        frame["market_structure_state"]
        == "TREND_EXPANSION_UP"
    )

    results = StructureAnalyzer(
        min_sample_size=3
    ).analyze(
        frame,
        condition,
        future_return_columns=(
            "future_return_1d",
            "future_return_5d",
        ),
        condition_name="TREND_EXPANSION_UP",
    )

    result = results["future_return_5d"]

    assert result.sample_size == 4
    assert result.mean_return == pytest.approx(0.035)
    assert result.win_rate == pytest.approx(0.75)
    assert result.median_return == pytest.approx(0.04)


def test_small_sample_is_not_reported_as_statistically_valid():

    frame = make_frame()

    condition = (
        frame["market_structure_state"]
        == "SIDEWAYS_ACCUMULATION"
    )

    results = StructureAnalyzer(
        min_sample_size=3
    ).analyze(
        frame,
        condition,
        future_return_columns=("future_return_5d",),
        condition_name="SIDEWAYS_ACCUMULATION",
    )

    result = results["future_return_5d"]

    assert result.sample_size == 1
    assert result.mean_return is None
    assert result.sharpe_ratio is None


def test_missing_future_return_column_raises():

    frame = make_frame()

    condition = pd.Series(
        [True] * len(frame),
        index=frame.index,
    )

    with pytest.raises(
        ValueError,
        match="missing required column",
    ):
        StructureAnalyzer().analyze(
            frame,
            condition,
            future_return_columns=("future_return_20d",),
        )


def test_condition_length_must_match():

    frame = make_frame()

    condition = pd.Series([True, False])

    with pytest.raises(
        ValueError,
        match="condition length",
    ):
        StructureAnalyzer().analyze(
            frame,
            condition,
            future_return_columns=("future_return_5d",),
        )
