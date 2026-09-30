from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StructureTransition:
    """单个时点发生的市场结构转移。"""

    previous_state: str
    current_state: str

    @property
    def label(self) -> str:
        return f"{self.previous_state}->{self.current_state}"


class MarketStructureTransitionDetector:
    """
    检测连续时点之间的市场结构状态变化。

    注意：
    1. 这里只描述状态转移。
    2. 不产生 BUY / SELL。
    3. 不使用未来数据。
    4. 不计算未来收益。
    """

    def __init__(
        self,
        state_column: str = "market_structure_state",
    ) -> None:
        if not state_column:
            raise ValueError("state_column must not be empty")

        self.state_column = state_column

    def detect(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.state_column not in frame.columns:
            raise ValueError(
                f"missing required column: {self.state_column}"
            )

        state = frame[self.state_column].astype("object")

        previous_state = state.shift(1)

        transition = pd.Series(
            "NO_TRANSITION",
            index=frame.index,
            dtype="object",
        )

        valid = (
            state.notna()
            & previous_state.notna()
            & state.ne(previous_state)
        )

        transition.loc[valid] = (
            previous_state.loc[valid].astype(str)
            + "->"
            + state.loc[valid].astype(str)
        )

        result = frame.copy()

        result["previous_market_structure_state"] = previous_state
        result["structure_transition"] = transition

        return result

    def detect_latest(
        self,
        frame: pd.DataFrame,
    ) -> StructureTransition | None:
        detected = self.detect(frame)

        if detected.empty:
            return None

        latest = detected.iloc[-1]

        if latest["structure_transition"] == "NO_TRANSITION":
            return None

        return StructureTransition(
            previous_state=str(
                latest["previous_market_structure_state"]
            ),
            current_state=str(
                latest[self.state_column]
            ),
        )
