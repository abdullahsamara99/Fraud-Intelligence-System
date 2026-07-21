from inference.decision_engine import DecisionEngine

engine = DecisionEngine()

scores = [
    -0.10,
    0.02,
    0.07,
    0.12,
    0.18,
]

for score in scores:

    decision = engine.evaluate(score)

    print(
        f"{score:>6} -> "
        f"{decision['risk_level']} -> "
        f"{decision['recommended_action']}"
    )