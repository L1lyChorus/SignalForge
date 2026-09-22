import pandas as pd
import pytest

from research.labels import (
    FutureReturnLabelBuilder,
    ResearchLabel,
)


def make_frame():
    return pd.DataFrame(
        {
            "close": [10, 11, 12, 13, 14, 15],
        }
    )


def test_research_label_validates():
    label = ResearchLabel(
        name="future_return_5d",
        horizon=5,
    )

    label.validate()


def test_research_label_rejects_invalid_horizon():
    label = ResearchLabel(
        name="future_return_5d",
        horizon=0,
    )

    with pytest.raises(
        ValueError,
        match="horizon must be greater than 0",
    ):
        label.validate()


def test_research_label_requires_name():
    label = ResearchLabel(
        name="",
        horizon=5,
    )

    with pytest.raises(
        ValueError,
        match="name is required",
    ):
        label.validate()


def test_builder_creates_future_return_label():
    result = FutureReturnLabelBuilder().build(
        make_frame(),
        horizon=2,
    )

    assert "future_return_2d" in result.columns

    assert result.loc[0, "future_return_2d"] == pytest.approx(
        12 / 10 - 1
    )

    assert result.loc[1, "future_return_2d"] == pytest.approx(
        13 / 11 - 1
    )


def test_builder_leaves_tail_without_future_data():
    result = FutureReturnLabelBuilder().build(
        make_frame(),
        horizon=2,
    )

    assert result["future_return_2d"].iloc[-1] != result["future_return_2d"].iloc[-1]
    assert result["future_return_2d"].iloc[-2] != result["future_return_2d"].iloc[-2]


def test_valid_mask_excludes_tail_rows():
    frame = make_frame()

    mask = FutureReturnLabelBuilder.valid_mask(
        frame,
        horizon=2,
    )

    assert list(mask) == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_builder_rejects_missing_price_column():
    frame = pd.DataFrame(
        {
            "open": [10, 11, 12],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required price column: close",
    ):
        FutureReturnLabelBuilder().build(
            frame,
            horizon=2,
        )


def test_builder_rejects_invalid_horizon():
    with pytest.raises(
        ValueError,
        match="horizon must be greater than 0",
    ):
        FutureReturnLabelBuilder().build(
            make_frame(),
            horizon=0,
        )
