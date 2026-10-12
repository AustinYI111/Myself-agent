import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.coordinator_agent import CoordinatorAgent


def test_coordinator_agent_basic():
    coordinator = CoordinatorAgent(
        agent_weights={
            "trend": 0.4,
            "mean_reversion": 0.3,
            "ml": 0.3
        }
    )

    agent_outputs = {
        "trend": {"signal": "buy", "confidence": 0.7},
        "mean_reversion": {"signal": "sell", "confidence": 0.6},
        "ml": {"signal": "buy", "confidence": 0.8}
    }

    result = coordinator.aggregate(agent_outputs)

    print("Coordinator output:", result)

    assert result["signal"] in ["buy", "sell", "hold"]
    assert 0 <= result["confidence"] <= 1


if __name__ == "__main__":
    test_coordinator_agent_basic()