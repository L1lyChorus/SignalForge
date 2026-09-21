import pandas as pd
import pytest

from research.validation import ResearchValidator


def make_frame():
    return pd.DataFrame(
        {
            "datetime": [
                "2021-01-01",
                "2021-01-02",
                "2021-01-03",
                "2021-01-04",
                "2021-01-05",
                "2022-01-01",
                "2022-01-02",
                "2022-01-03",
                "2022-01-04",
                "2022-01-05",
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "signal": [
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
                True,
                False,
                True,
                False,
                True,
            ],
            "future_return_5d": [
                0.10,
                0.01,
                0.08,
                -0.01,
                0.06,
                0.09,
                0.02,
                0.07,
                -0.01,
                0.05,
                0.08,
                0.01,
                0.06,
                -0.02,
                0.05,
            ],
        }
    )


def test_validation_runs_across_three_periods():
    result = ResearchValidator().validate(
        make_frame(),
        lambda data: data["signal"],
        train_end="2021-12-31",
        validation_end="2022-12-31",
    )

    assert result.in_sample.sample_size == 3
    assert result.validation.sample_size == 3
    assert result.out_of_sample.sample_size == 3

    assert result.in_sample.mean_return > 0
    assert result.validation.mean_return > 0
    assert result.out_of_sample.mean_return > 0

    assert result.stable is True


def test_validation_calculates_decay():
    result = ResearchValidator().validate(
        make_frame(),
        lambda data: data["signal"],
        train_end="2021-12-31",
        validation_end="2022-12-31",
    )

    expected_decay = (
        result.out_of_sample.mean_return
        - result.in_sample.mean_return
    )

    assert result.mean_return_decay == pytest.approx(
        expected_decay
    )

    expected_win_rate_decay = (
        result.out_of_sample.win_rate
        - result.in_sample.win_rate
    )

    assert result.win_rate_decay == pytest.approx(
        expected_win_rate_decay
    )
