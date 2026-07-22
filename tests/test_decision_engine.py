import pytest

from inference.decision_engine import DecisionEngine


@pytest.fixture
def engine():
    return DecisionEngine()


def test_high_risk_anomaly(engine):
    result = engine.evaluate(
        prediction=-1,
        risk_score=0.40,
    )

    assert result["risk_level"] == "High"
    assert result["recommended_action"] == "Block Transaction"


def test_high_risk_score(engine):
    result = engine.evaluate(
        prediction=1,
        risk_score=0.90,
    )

    assert result["risk_level"] == "High"


def test_medium_risk(engine):
    result = engine.evaluate(
        prediction=1,
        risk_score=0.65,
    )

    assert result["risk_level"] == "Medium"
    assert result["recommended_action"] == "Manual Review"


def test_low_risk(engine):
    result = engine.evaluate(
        prediction=1,
        risk_score=0.20,
    )

    assert result["risk_level"] == "Low"
    assert result["recommended_action"] == "Approve"