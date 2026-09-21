import pandas as pd
import pytest

from research.purged_split import PurgedResearchDataSplitter


def make_frame():
    return pd.DataFrame(
        {
            "datetime": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
                "2023-01-06",
                "2023-01-07",
                "2023-01-08",
                "2023-01-09",
                "2023-01-10",
            ],
            "close": range(10, 20),
        }
    )


def test_purged_split_removes_boundary_rows():
    result = PurgedResearchDataSplitter().split(
        make_frame(),
        train_end="2023-01-05",
        validation_end="2023-01-10",
        purge_days=2,
    )

    assert list(result.in_sample["close"]) == [10, 11, 12, 13, 14]
    assert list(result.purged["close"]) == [15, 16]
    assert list(result.validation["close"]) == [17, 18, 19]
    assert list(result.out_of_sample["close"]) == []


def test_purged_split_preserves_all_rows():
    result = PurgedResearchDataSplitter().split(
        make_frame(),
        train_end="2023-01-05",
        validation_end="2023-01-08",
        purge_days=2,
    )

    total_rows = (
        len(result.in_sample)
        + len(result.purged)
        + len(result.validation)
        + len(result.out_of_sample)
    )

    assert total_rows == len(make_frame())


def test_purged_split_has_no_overlap():
    result = PurgedResearchDataSplitter().split(
        make_frame(),
        train_end="2023-01-05",
        validation_end="2023-01-08",
        purge_days=2,
    )

    groups = [
        set(result.in_sample["datetime"]),
        set(result.purged["datetime"]),
        set(result.validation["datetime"]),
        set(result.out_of_sample["datetime"]),
    ]

    for index, group in enumerate(groups):
        for other in groups[index + 1:]:
            assert group.isdisjoint(other)


def test_zero_purge_matches_normal_boundary_behavior():
    result = PurgedResearchDataSplitter().split(
        make_frame(),
        train_end="2023-01-05",
        validation_end="2023-01-08",
        purge_days=0,
    )

    assert list(result.in_sample["close"]) == [10, 11, 12, 13, 14]
    assert result.purged.empty
    assert list(result.validation["close"]) == [15, 16, 17]
    assert list(result.out_of_sample["close"]) == [18, 19]


def test_rejects_negative_purge_days():
    with pytest.raises(
        ValueError,
        match="purge_days must be greater than or equal to 0",
    ):
        PurgedResearchDataSplitter().split(
            make_frame(),
            train_end="2023-01-05",
            validation_end="2023-01-08",
            purge_days=-1,
        )


def test_rejects_invalid_boundaries():
    with pytest.raises(
        ValueError,
        match="train_end must be before validation_end",
    ):
        PurgedResearchDataSplitter().split(
            make_frame(),
            train_end="2023-01-08",
            validation_end="2023-01-05",
            purge_days=2,
        )
