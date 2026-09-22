import pandas as pd
import pytest

from research.analyzer import ResearchAnalyzer
from research.report import ResearchReport
from research.robustness import ResearchRobustnessAnalyzer


def make_analysis():
    frame = pd.DataFrame(
        {
            "future_return_5d": [
                0.10,
                0.05,
                -0.02,
                0.03,
            ]
        }
    )

    return ResearchAnalyzer().analyze(
        frame,
        lambda data: pd.Series(True, index=data.index),
    )


def test_report_requires_hypothesis_id():
    report = ResearchReport(
        hypothesis_id="",
        hypothesis_title="测试假设",
        in_sample=make_analysis(),
    )

    with pytest.raises(
        ValueError,
        match="hypothesis_id is required",
    ):
        report.validate()


def test_report_requires_hypothesis_title():
    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="",
        in_sample=make_analysis(),
    )

    with pytest.raises(
        ValueError,
        match="hypothesis_title is required",
    ):
        report.validate()


def test_report_requires_at_least_one_analysis():
    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="测试假设",
    )

    with pytest.raises(
        ValueError,
        match="at least one research analysis",
    ):
        report.validate()


def test_report_exposes_out_of_sample_metrics():
    analysis = make_analysis()

    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="测试假设",
        out_of_sample=analysis,
    )

    report.validate()

    assert report.out_of_sample_mean_return == pytest.approx(
        analysis.mean_return
    )

    assert report.out_of_sample_sharpe == pytest.approx(
        analysis.sharpe_ratio
    )


def test_report_exposes_robustness_ratio():
    frame = pd.DataFrame(
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

    robustness = ResearchRobustnessAnalyzer().analyze(
        frame,
        {
            "threshold_30": (
                lambda data: data["volume_change"] > 0.3
            ),
            "threshold_50": (
                lambda data: data["volume_change"] > 0.5
            ),
        },
        min_sample_size=2,
    )

    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="放量上涨后的短期延续",
        in_sample=make_analysis(),
        robustness=robustness,
    )

    report.validate()

    assert report.stability_ratio == pytest.approx(1.0)


def test_report_validates_purged_sample_size():
    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="测试假设",
        in_sample=make_analysis(),
        purged_sample_size=3,
    )

    report.validate()

    assert report.purged_sample_size == 3


def test_report_rejects_negative_purged_sample_size():
    report = ResearchReport(
        hypothesis_id="H001",
        hypothesis_title="测试假设",
        in_sample=make_analysis(),
        purged_sample_size=-1,
    )

    with pytest.raises(
        ValueError,
        match="purged_sample_size must be greater than or equal to 0",
    ):
        report.validate()
