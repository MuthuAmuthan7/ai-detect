import pytest
from app.inference.decision import DecisionEngine


def test_decision_engine_ema_and_hysteresis():
    engine = DecisionEngine(
        ema_alpha=0.4,
        ai_threshold=0.90,
        human_threshold=0.10,
        ai_confirmations=2,
        human_confirmations=3
    )

    # 1. First high score -> EMA updates, but confirmation counter is 1 < 2 -> status 'unknown'
    status, ema = engine.update(0.95)
    assert ema == 0.95
    assert status == "unknown"

    # 2. Second high score -> EMA >= 0.90, confirmation counter reaches 2 -> status 'ai'
    status, ema = engine.update(0.95)
    assert ema >= 0.90
    assert status == "ai"

    # 3. Sudden low human score -> EMA drops, AI counter resets -> status reverts to 'unknown' until 3 human confirmations
    status, ema = engine.update(0.05)
    assert status == "unknown"


def test_decision_engine_human_confirmation():
    engine = DecisionEngine(
        ema_alpha=0.5,
        ai_threshold=0.90,
        human_threshold=0.10,
        ai_confirmations=2,
        human_confirmations=3
    )

    engine.update(0.02)
    engine.update(0.02)
    status, ema = engine.update(0.02)

    assert ema <= 0.10
    assert status == "human"
