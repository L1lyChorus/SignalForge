import pandas as pd
import pytest

from research.split import ResearchDataSplitter


def make_frame():
    return pd.DataFrame(
        {
            "datetime": [
                "2023-01-03",
                "2024-01-03",
                "2025-01-03",
                "2022-01-03",
                "2026-01-03",
            ],
            "close": [10, 11, 12, 9, 13],
        }
    )


def test_split_is_time_ordered_and_disjoint():
    result = ResearchDataSplitter().split(
        make_frame(),
        train_end="2023-12-31",
        validation_end="2024-12-31",
    )

    assert list(result.in_sample["close"]) == [9, 10]
    assert list(result.validation["close"]) == [11]
    assert list(result.out_of_sample["close"]) == [12, 13]


def test_split_preserves_all_rows_without_overlap():
    result = ResearchDataSplitter().split(
        make_frame(),
        train_end="2023-12-31",
        validation_end="2024-12-31",
    )

    total_rows = (
        len(result.in_sample)
        + len(result.validation)
        + len(result.out_of_sample)
    )

    assert total_rows == len(make_frame())

    in_sample_dates = set(result.in_sample["datetime"])
    validation_dates = set(result.validation["datetime"])
    out_of_sample_dates = set(result.out_of_sample["datetime"])

    assert in_sample_dates.isdisjoint(validation_dates)
    assert in_sample_dates.isdisjoint(out_of_sample_dates)
    assert validation_dates.isdisjoint(out_of_sample_dates)


def test_split_requires_datetime():
    frame = pd.DataFrame({"close": [10, 11]})

    with pytest.raises(
        ValueError,
        match="missing required column: datetime",
    ):
        ResearchDataSplitter().split(
            frame,
            train_end="2023-12-31",
            validation_end="2024-12-31",
        )


def test_split_rejects_empty_frame():
    frame = pd.DataFrame(
        {
            "datetime": pd.Series(dtype="datetime64[ns]"),
            "close": pd.Series(dtype=float),
        }
    )

    with pytest.raises(
        ValueError,
        match="frame must not be empty",
    ):
        ResearchDataSplitter().split(
            frame,
            train_end="2023-12-31",
            validation_end="2024-12-31",
        )


def test_split_rejects_invalid_boundaries():
    frame = make_frame()

    with pytest.raises(
        ValueError,
        match="train_end must be before validation_end",
    ):
        ResearchDataSplitter().split(
            frame,
            train_end="2025-01-01",
            validation_end="2024-01-01",
        )


def test_split_rejects_invalid_datetime_boundary():
    frame = make_frame()

    with pytest.raises(
        ValueError,
        match="train_end must be a valid datetime",
    ):
        ResearchDataSplitter().split(
            frame,
            train_end="not-a-date",
            validation_end="2024-12-31",
        )


def test_split_rejects_invalid_frame_datetime():
    frame = pd.DataFrame(
        {
            "datetime": [
                "2023-01-01",
                "not-a-date",
            ],
            "close": [10, 11],
        }
    )

    with pytest.raises(
        ValueError,
        match="datetime contains invalid values",
    ):
        ResearchDataSplitter().split(
            frame,
            train_end="2023-12-31",
            validation_end="2024-12-31",
        )
