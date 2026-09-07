from typing import Literal, Tuple
from collections import deque
from app.config import settings

StatusType = Literal["ai", "human", "unknown"]


class DecisionEngine:
    """
    Rolling Exponential Moving Average (EMA) probability aggregator & Hysteresis State Machine.
    Ensures predictions are smoothed across 3-second sliding windows and prevents rapid flipping.
    """

    def __init__(
        self,
        ema_alpha: float = settings.EMA_ALPHA,
        ai_threshold: float = settings.AI_THRESHOLD,
        human_threshold: float = settings.HUMAN_THRESHOLD,
        ai_confirmations: int = settings.AI_CONFIRMATIONS,
        human_confirmations: int = settings.HUMAN_CONFIRMATIONS
    ):
        self.ema_alpha = ema_alpha
        self.ai_threshold = ai_threshold
        self.human_threshold = human_threshold
        self.ai_confirmations = ai_confirmations
        self.human_confirmations = human_confirmations

        self.ema_score: float = 0.5  # Neutral starting score
        self.history: deque = deque(maxlen=10)
        self.current_status: StatusType = "unknown"

        self.consecutive_ai_count: int = 0
        self.consecutive_human_count: int = 0
        self.total_predictions: int = 0

    def update(self, raw_ai_score: float) -> Tuple[StatusType, float]:
        """
        Updates internal EMA and hysteresis state with new raw AI score.
        Returns:
            Tuple[current_status, smoothed_ema_score]
        """
        self.total_predictions += 1
        self.history.append(raw_ai_score)

        if self.total_predictions == 1:
            self.ema_score = raw_ai_score
        else:
            self.ema_score = self.ema_alpha * raw_ai_score + (1.0 - self.ema_alpha) * self.ema_score

        # Check threshold candidates against smoothed score
        if self.ema_score >= self.ai_threshold:
            self.consecutive_ai_count += 1
            self.consecutive_human_count = 0
        elif self.ema_score <= self.human_threshold:
            self.consecutive_human_count += 1
            self.consecutive_ai_count = 0
        else:
            self.consecutive_ai_count = 0
            self.consecutive_human_count = 0

        # State transition logic
        if self.consecutive_ai_count >= self.ai_confirmations:
            self.current_status = "ai"
        elif self.consecutive_human_count >= self.human_confirmations:
            self.current_status = "human"
        else:
            if self.consecutive_ai_count == 0 and self.consecutive_human_count == 0:
                self.current_status = "unknown"

        return self.current_status, round(self.ema_score, 4)

    def reset(self):
        """Resets decision engine state for a new call."""
        self.ema_score = 0.5
        self.history.clear()
        self.current_status = "unknown"
        self.consecutive_ai_count = 0
        self.consecutive_human_count = 0
        self.total_predictions = 0
