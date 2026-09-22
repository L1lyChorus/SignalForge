import pandas as pd

from research.experiment import ResearchExperiment
from research.ledger import ResearchLedger
from research.models import ResearchHypothesis
from research.runner import ResearchExperimentRunner


def make_hypothesis():
    return ResearchHypothesis(
        hypothesis_id="H001",
        title="放量上涨后的短期延续",
        economic_reason="成交量显著增加可能反映市场参与度和资金关注度上升。",
        variables=["return_1d", "volume_change", "future_return_5d"],
        expected_effect="放量上涨后未来5日收益率可能更高。",
        research_method="按成交量变化分组，比较不同组的未来收益。",
    )


def make_frame():
    return pd.DataFrame(
        {
            "open": [10, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "high": [11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            "low": [9, 9, 10, 11, 12, 13, 14, 15, 16, 17],
            "close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            "volume": [100, 180, 200, 150, 250, 300, 320, 330, 340, 350],
        }
    )


def make_experiment():
    return ResearchExperiment(
        experiment_id="EXP001",
        hypothesis_id="H001",
        condition_name="volume_change_gt_40pct",
        data_source="test_market_data",
        start_date="2020-01-01",
        end_date="2025-12-31",
    )


def test_runner_executes_and_records_result():
    ledger = ResearchLedger()

    ledger.register_hypothesis(make_hypothesis())
    ledger.register_experiment(make_experiment())

    runner = ResearchExperimentRunner(ledger)

    result = runner.run(
        make_experiment(),
        make_frame(),
        lambda data: data["volume_change"] > 0.4,
    )

    assert result.experiment_id == "EXP001"
    assert result.hypothesis_id == "H001"
    assert result.sample_size == 2

    stored_results = ledger.get_results("H001")

    assert len(stored_results) == 1
    assert stored_results[0].experiment_id == "EXP001"


def test_runner_builds_research_report():
    ledger = ResearchLedger()

    hypothesis = make_hypothesis()
    experiment = make_experiment()

    ledger.register_hypothesis(hypothesis)
    ledger.register_experiment(experiment)

    runner = ResearchExperimentRunner(ledger)

    analysis = runner.pipeline.analyzer.analyze(
        runner.pipeline.prepare_features(make_frame()),
        lambda data: data["volume_change"] > 0.4,
    )

    report = runner.build_report(
        experiment=experiment,
        hypothesis_title=hypothesis.title,
        in_sample=analysis,
    )

    assert report.hypothesis_id == "H001"
    assert report.hypothesis_title == hypothesis.title
    assert report.in_sample is analysis
    assert report.out_of_sample is None


def test_runner_result_preserves_sharpe_ratio():
    ledger = ResearchLedger()

    ledger.register_hypothesis(make_hypothesis())
    ledger.register_experiment(make_experiment())

    runner = ResearchExperimentRunner(ledger)

    result = runner.run(
        make_experiment(),
        make_frame(),
        lambda data: data["volume_change"] > 0.4,
    )

    assert result.sharpe_ratio is not None


def make_validation_frame():
    return pd.DataFrame(
        {
            "datetime": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
                "2020-01-06",
                "2020-01-07",
                "2020-01-08",
                "2020-01-09",
                "2020-01-10",
                "2020-01-11",
                "2020-01-12",
                "2020-01-13",
                "2020-01-14",
                "2020-01-15",
                "2020-01-16",
                "2020-01-17",
                "2020-01-18",
                "2020-01-19",
                "2020-01-20",
            ],
            "open": list(range(10, 30)),
            "high": list(range(11, 31)),
            "low": list(range(9, 29)),
            "close": list(range(10, 30)),
            "volume": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
                180,
                190,
                200,
                210,
                220,
                230,
                240,
                250,
                260,
                270,
                280,
                290,
            ],
        }
    )


def test_runner_run_validation_returns_three_stage_report():
    ledger = ResearchLedger()

    hypothesis = make_hypothesis()
    experiment = make_experiment()

    ledger.register_hypothesis(hypothesis)
    ledger.register_experiment(experiment)

    runner = ResearchExperimentRunner(ledger)

    report = runner.run_validation(
        experiment,
        make_validation_frame(),
        lambda data: pd.Series(True, index=data.index),
        train_end="2020-01-07",
        validation_end="2020-01-14",
        purge_days=1,
    )

    assert report.hypothesis_id == "H001"
    assert report.hypothesis_title == hypothesis.title

    assert report.in_sample is not None
    assert report.validation is not None
    assert report.out_of_sample is not None

    assert report.purged_sample_size == 1


def test_runner_run_validation_keeps_time_windows_disjoint():
    ledger = ResearchLedger()

    ledger.register_hypothesis(make_hypothesis())
    ledger.register_experiment(make_experiment())

    runner = ResearchExperimentRunner(ledger)

    report = runner.run_validation(
        make_experiment(),
        make_validation_frame(),
        lambda data: pd.Series(True, index=data.index),
        train_end="2020-01-07",
        validation_end="2020-01-14",
        purge_days=2,
    )

    assert report.in_sample.sample_size == 7
    assert report.validation.sample_size == 5
    assert report.out_of_sample.sample_size == 1
    assert report.purged_sample_size == 2


def test_runner_run_validation_uses_registered_hypothesis_title():
    ledger = ResearchLedger()

    hypothesis = make_hypothesis()
    experiment = make_experiment()

    ledger.register_hypothesis(hypothesis)
    ledger.register_experiment(experiment)

    runner = ResearchExperimentRunner(ledger)

    report = runner.run_validation(
        experiment,
        make_validation_frame(),
        lambda data: pd.Series(True, index=data.index),
        train_end="2020-01-07",
        validation_end="2020-01-14",
    )

    assert report.hypothesis_title == hypothesis.title
