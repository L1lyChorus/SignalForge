import pandas as pd
import pytest

from research.regime.volume_price import (
    VolumePriceState,
    VolumePriceStructureDetector,
)


def make_frame(
    close,
    volume,
    high=None,
    low=None,
):
    close = list(close)
    volume = list(volume)

    if high is None:
        high = [value * 1.02 for value in close]

    if low is None:
        low = [value * 0.98 for value in close]

    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_missing_required_columns_are_rejected():
    detector = VolumePriceStructureDetector()

    frame = pd.DataFrame(
        {
            "close": [10, 11, 12],
            "volume": [100, 100, 100],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        detector.detect(frame)


def test_initial_rows_are_unknown():
    detector = VolumePriceStructureDetector(
        volume_window=3,
    )

    frame = make_frame(
        close=[10, 10, 10, 11],
        volume=[100, 100, 100, 100],
    )

    result = detector.detect(frame)

    assert result.iloc[0] == VolumePriceState.UNKNOWN.value
    assert result.iloc[1] == VolumePriceState.UNKNOWN.value


def test_volume_expansion_with_price_breakout_is_expansion_up():
    detector = VolumePriceStructureDetector(
        volume_window=3,
        expansion_threshold=2.0,
        price_threshold=0.01,
    )

    frame = make_frame(
        close=[10, 10, 10, 10.5],
        volume=[100, 100, 100, 250],
        high=[10.2, 10.2, 10.2, 10.6],
        low=[9.8, 9.8, 9.8, 10.0],
    )

    result = detector.detect(frame)

    assert result.iloc[-1] == (
        VolumePriceState.EXPANSION_UP.value
    )


def test_volume_expansion_with_price_breakdown_is_expansion_down():
    detector = VolumePriceStructureDetector(
        volume_window=3,
        expansion_threshold=2.0,
        price_threshold=0.01,
    )

    frame = make_frame(
        close=[10, 10, 10, 9.5],
        volume=[100, 100, 100, 250],
        high=[10.2, 10.2, 10.2, 9.8],
        low=[9.8, 9.8, 9.8, 9.4],
    )

    result = detector.detect(frame)

    assert result.iloc[-1] == (
        VolumePriceState.EXPANSION_DOWN.value
    )


def test_low_relative_volume_is_contraction():
    detector = VolumePriceStructureDetector(
        volume_window=3,
        contraction_threshold=0.7,
    )

    frame = make_frame(
        close=[10, 10, 10, 10.01],
        volume=[100, 100, 100, 50],
    )

    result = detector.detect(frame)

    assert result.iloc[-1] == (
        VolumePriceState.CONTRACTION.value
    )


def test_flat_price_with_large_volume_and_strong_close_is_accumulation():
    detector = VolumePriceStructureDetector(
        volume_window=3,
        expansion_threshold=2.0,
        price_threshold=0.01,
    )

    frame = make_frame(
        close=[10, 10, 10, 10.03],
        volume=[100, 100, 100, 250],
        high=[10.2, 10.2, 10.2, 10.05],
        low=[9.8, 9.8, 9.8, 9.7],
    )

    result = detector.detect(frame)

    assert result.iloc[-1] == (
        VolumePriceState.ACCUMULATION.value
    )


def test_flat_price_with_large_volume_and_weak_close_is_distribution():
    detector = VolumePriceStructureDetector(
        volume_window=3,
        expansion_threshold=2.0,
        price_threshold=0.01,
    )

    frame = make_frame(
        close=[10, 10, 10, 9.97],
        volume=[100, 100, 100, 250],
        high=[10.3, 10.3, 10.3, 10.3],
        low=[9.8, 9.8, 9.8, 9.9],
    )

    result = detector.detect(frame)

    assert result.iloc[-1] == (
        VolumePriceState.DISTRIBUTION.value
    )
