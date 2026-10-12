import pandas as pd
from agents.trend_agent import TrendAgent

data = pd.DataFrame({
    "close": [10,11,12,13,14,15,16,17,18,19,20]
})

agent = TrendAgent()
print(agent.generate_signal(data))
