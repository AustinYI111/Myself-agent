from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker


@dataclass(frozen=True)
class Config:
    train_horizon: int = 6
    top_k: int = 20
    train_months: int = 60
    min_train_groups: int = 12
    min_group_size: int = 20
    embargo_sessions: int = 1
    oneil: bool = False
    commission_bps: float = 3.0
    slippage_bps: float = 5.0
    sell_tax_bps: float = 5.0  # research assumption; user must supply dated rates
    n_estimators: int = 60
    seed: int = 42

    def __post_init__(self):
        if self.train_horizon not in [3, 6, 12]:
            raise ValueError("train_horizon must be 3/6/12")
        for name in ["top_k", "train_months", "min_train_groups", "min_group_size", "n_estimators"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.embargo_sessions, int) or isinstance(self.embargo_sessions, bool) or self.embargo_sessions < 0:
            raise ValueError("embargo must be a nonnegative integer")
        costs = [self.commission_bps, self.slippage_bps, self.sell_tax_bps]
        if not np.isfinite(costs).all() or min(costs) < 0 or sum(costs) >= 10000:
            raise ValueError("costs must be finite, nonnegative and total below 10000 bps")


def training_slice(samples, date, calendar, config):
    position = calendar.get_indexer([date])[0]
    if position < 0:
        raise ValueError("prediction date outside calendar")
    cutoff = calendar[max(0, position - config.embargo_sessions)]
    # Purge by actual label end, never by feature date alone.
    train = samples[(samples.date < date) & (samples.label_end < cutoff)
                    & (samples.date >= date - pd.DateOffset(months=config.train_months))
                    & samples.target.notna()].copy()
    sizes = train.groupby("date").size()
    train = train[train.date.isin(sizes[sizes >= config.min_group_size].index)]
    return train.sort_values(["date", "symbol"]), cutoff


def fit_ranker(train, features, config):
    # Five ordinal relevance levels within each monthly query. Preserve ties.
    relevance = train.groupby("date").target.rank(method="average", pct=True)
    y = np.minimum((relevance * 5).astype(int), 4)
    groups = train.groupby("date", sort=True).size().to_numpy()
    model = LGBMRanker(objective="lambdarank", n_estimators=config.n_estimators,
                       learning_rate=0.05, num_leaves=15, max_depth=4,
                       min_child_samples=15, reg_lambda=2.0, verbosity=-1,
                       random_state=config.seed, n_jobs=1, deterministic=True,
                       force_col_wise=True, importance_type="gain")
    model.fit(train[features], y, group=groups)
    return model
