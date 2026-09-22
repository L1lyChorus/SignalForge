import pandas as pd
import pytest

from research.robustness import ResearchRobustnessAnalyzer


def make_frame():
    return pd.DataFrame(
        {
            "volume_change": [
                0.1,
                0.3,
                0.5,
                0.7,
                1.0,
                1.5,
            ],
            "future_return_5d": [
                -0.02,
                0.01,
                0.04,
                0.06,
                0.08,
                0.10,
            ],
        }
    )


def test_robustness_runs_multiple_conditions():
    result = ResearchRobustnessAnalyzer().analyze(
        make_frame(),
        {
            "threshold_30": lambda data: data["volume_change"] > 0.3,
            "threshold_50": lambda data: data["volume_change"] > 0.5,
            "threshold_70": lambda data: data["volume_change"] > 0.7,
        },
        min_sample_size=2,
    )

    assert result.total_case_count == 3
    assert result.valid_case_count == 3
    assert result.stability_ratio == pytest.approx(1.0)


def test_small_sample_is_not_valid():
    result = ResearchRobustnessAnalyzer().analyze(
        make_frame(),
        {
            "threshold_30": lambda data: data["volume_change"] > 0.3,
        },
        min_sample_size=10,
    )

    assert result.total_case_count == 1
    assert result.valid_case_count == 0
    assert result.cases[0].sample_size_valid is False
    assert result.cases[0].return_valid is True
    assert result.cases[0].sharpe_valid is True


def test_negative_return_is_not_valid():
    frame = pd.DataFrame(
        {
            "signal": [True, True, True, True],
            "future_return_5d": [
                -0.01,
                -0.02,
                -0.03,
                -0.04,
            ],
        }
    )

    result = ResearchRobustnessAnalyzer().analyze(
        frame,
        {
            "negative_case": lambda data: data["signal"],
        },
        min_sample_size=2,
    )

    case = result.cases[0]

    assert case.sample_size_valid is True
    assert case.return_valid is False
    assert case.is_valid is False
    assert result.valid_case_count == 0


def test_robustness_requires_conditions():
    with pytest.raises(
        ValueError,
        match="conditions must not be empty",
    ):
        ResearchRobustnessAnalyzer().analyze(
            make_frame(),
            {},
        )


def test_robustness_rejects_invalid_sample_threshold():
    with pytest.raises(
        ValueError,
        match="min_sample_size must be greater than 0",
    ):
        ResearchRobustnessAnalyzer().analyze(
            make_frame(),
            {
                "threshold": lambda data: data["volume_change"] > 0.3,
            },
            min_sample_size=0,
        )
