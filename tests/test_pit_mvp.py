from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ashare_mvp.data import validate, fundamentals_asof, month_ends
from ashare_mvp.demo import make_demo, save
from ashare_mvp.data import load
from ashare_mvp.factors import default_library, technical_panel, snapshot
from ashare_mvp.model import Config, training_slice, fit_ranker
from ashare_mvp.pipeline import build_samples, forward_labels, rolling_rank, simulate_monthly, cohort_performance, run, nav_metrics, _finite_mean


@pytest.fixture(scope="module")
def data():
    return make_demo(stocks=24, periods=1050)


def test_duplicate_prices_rejected(data):
    with pytest.raises(ValueError, match="duplicate price"):
        validate(replace(data, prices=pd.concat([data.prices, data.prices.iloc[:1]])))


def test_missing_session_rejected(data):
    with pytest.raises(ValueError, match="incomplete calendar"):
        validate(replace(data, prices=data.prices.iloc[1:]))


def test_availability_rejected(data):
    f = data.fundamentals.copy()
    f.loc[0, "available_date"] = f.loc[0, "report_date"]
    with pytest.raises(ValueError, match="available after"):
        validate(replace(data, fundamentals=f))


def test_revision_and_old_period_correction(data):
    symbol = data.prices.symbol.iloc[0]
    f = data.fundamentals[data.fundamentals.symbol == symbol].iloc[:2].copy()
    original = f.iloc[0].copy()
    correction = original.copy()
    correction["available_date"] = original.available_date + pd.Timedelta(days=10)
    correction["roe"] = 999
    late_old = correction.copy()
    late_old["available_date"] = f.iloc[1].available_date + pd.Timedelta(days=10)
    d = replace(data, fundamentals=pd.concat([f, pd.DataFrame([correction, late_old])]))
    assert fundamentals_asof(d, original.available_date).iloc[0].roe == original.roe
    assert fundamentals_asof(d, correction.available_date).iloc[0].roe == 999
    assert fundamentals_asof(d, late_old.available_date).iloc[0].report_date == f.iloc[1].report_date


def test_unannounced_reports_unavailable(data):
    f = data.fundamentals.iloc[:1]
    d = replace(data, fundamentals=f)
    assert fundamentals_asof(d, f.iloc[0].available_date - pd.Timedelta(days=1)).empty


def test_optional_oneil_and_extension(data):
    date = month_ends(data.calendar)[20]
    snap = snapshot(data, technical_panel(data), date)
    lib = default_library()
    assert len(lib.compute(snap).columns) == 10
    assert len(lib.compute(snap, oneil=True).columns) == 13
    lib.register("test_factor", "custom", lambda x: x.roe * 2)
    assert "test_factor" in lib.compute(snap)
    with pytest.raises(ValueError, match="duplicate factor"):
        lib.register("test_factor", "custom", lambda x: x.roe)


def test_historical_factors_immune_to_future(data):
    date = month_ends(data.calendar)[20]
    prices = data.prices.copy()
    prices.loc[prices.date > date, ["close", "volume", "market_cap"]] *= 10
    f = data.fundamentals.copy()
    f.loc[f.available_date > date, "roe"] = 999
    altered = replace(data, prices=prices, fundamentals=f)
    lib = default_library()
    left = lib.compute(snapshot(data, technical_panel(data), date), True)
    right = lib.compute(snapshot(altered, technical_panel(altered), date), True)
    pd.testing.assert_frame_equal(left, right)


def test_calendar_horizon_and_next_open(data):
    date = month_ends(data.calendar)[20]
    labels = forward_labels(data, [date])
    symbol = data.prices.symbol.iloc[0]
    row = labels[(labels.symbol == symbol) & (labels.horizon == 3)].iloc[0]
    entry = data.calendar[data.calendar.searchsorted(date, side="right")]
    end = data.calendar[data.calendar.searchsorted(entry + pd.DateOffset(months=3))]
    assert row.entry_date == entry and row.label_end == end
    p = data.prices[data.prices.symbol == symbol].set_index("date")
    assert row.target == pytest.approx(p.at[end, "total_return_open"] / p.at[entry, "total_return_open"] - 1)


def test_unmatured_labels_not_shortened(data):
    labels = forward_labels(data, [data.calendar[-10]])
    assert labels.target.isna().all()
    assert labels.label_end.isna().all()


def test_training_purges_label_end(data):
    cfg = Config(min_train_groups=3, n_estimators=5)
    samples, _, features = build_samples(data, cfg)
    date = samples.date.max()
    train, cutoff = training_slice(samples, date, data.calendar, cfg)
    assert not train.empty
    assert (train.label_end < cutoff).all()
    assert (train.date < date).all()
    assert not set(features).intersection({"target", "label_end", "entry_date", "available_date", "date"})
    assert len(samples[(samples.date < date) & (samples.label_end >= cutoff)]) > 0


def test_real_lightgbm_fit_is_repeatable(data):
    cfg = Config(n_estimators=8)
    samples, _, features = build_samples(data, cfg)
    train, _ = training_slice(samples, samples.date.max(), data.calendar, cfg)
    a, b = fit_ranker(train, features, cfg), fit_ranker(train, features, cfg)
    np.testing.assert_array_equal(a.predict(train[features]), b.predict(train[features]))
    assert a.booster_.num_trees() > 0


def test_rankings_immune_to_future_outcomes(data):
    cfg = Config(min_train_groups=4, n_estimators=5)
    date = month_ends(data.calendar)[35]
    dates = month_ends(data.calendar)
    dates = dates[dates <= date]
    samples, _, features = build_samples(data, cfg, dates)
    train, _ = training_slice(samples, date, data.calendar, cfg)
    a = fit_ranker(train, features, cfg)
    prices = data.prices.copy()
    prices.loc[prices.date > date, "total_return_open"] *= 4
    altered = replace(data, prices=prices)
    new_samples, _, _ = build_samples(altered, cfg, dates)
    new_train, _ = training_slice(new_samples, date, data.calendar, cfg)
    b = fit_ranker(new_train, features, cfg)
    pd.testing.assert_frame_equal(train, new_train)
    current = samples[samples.date == date]
    np.testing.assert_array_equal(a.predict(current[features]), b.predict(current[features]))


def test_st_excluded_at_snapshot(data):
    p = data.prices.copy()
    date = month_ends(data.calendar)[20]
    symbol = p.symbol.iloc[0]
    p.loc[(p.date == date) & (p.symbol == symbol), "is_st"] = True
    d = replace(data, prices=p)
    assert symbol not in snapshot(d, technical_panel(d), date).index


def test_blocked_buy_stays_cash_and_costs_count(data):
    date = month_ends(data.calendar)[20]
    entry = data.calendar[data.calendar.searchsorted(date, side="right")]
    symbol = data.prices.symbol.iloc[0]
    picks = pd.DataFrame([dict(date=date, symbol=symbol, target_weight=1.0)])
    p = data.prices.copy()
    p.loc[(p.date == entry) & (p.symbol == symbol), "can_buy"] = False
    nav, orders = simulate_monthly(replace(data, prices=p), picks, Config())
    assert (nav.nav == 1).all() and orders.status.iloc[0] == "blocked"
    nav, orders = simulate_monthly(data, picks, Config())
    assert nav.nav.iloc[0] < 1
    assert orders.cost.sum() == pytest.approx(1 - nav.nav.iloc[0])
    assert nav.cash.min() >= -1e-9


def test_blocked_sale_preserves_position(data):
    dates = month_ends(data.calendar)[20:22]
    symbols = sorted(data.prices.symbol.unique())[:2]
    picks = pd.DataFrame([dict(date=d, symbol=s, target_weight=1.0) for d, s in zip(dates, symbols)])
    entry = data.calendar[data.calendar.searchsorted(dates[1], side="right")]
    p = data.prices.copy()
    p.loc[(p.date == entry) & (p.symbol == symbols[0]), "can_sell"] = False
    nav, orders = simulate_monthly(replace(data, prices=p), picks, Config())
    assert ((orders.side == "sell") & (orders.status == "blocked")).any()
    assert nav[nav.date == entry].positions.iloc[0] == 1


def test_no_partial_cohort_mean(data):
    cfg = Config(top_k=20)
    samples, labels, _ = build_samples(data, cfg)
    date = samples.date.max()
    selected = samples[samples.date == date].head(20)[["date", "symbol"]]
    result = cohort_performance(selected, labels, samples, data, cfg)
    assert not result.complete.any()
    assert result.gross_return.isna().all()


def test_roundtrip_csv(data, tmp_path):
    save(data, tmp_path / "input")
    restored = load(tmp_path / "input")
    assert len(restored.prices) == len(data.prices)
    assert restored.metadata["synthetic"] is True


def test_full_pipeline_asof_and_outputs(data, tmp_path):
    cfg = Config(min_train_groups=4, n_estimators=5)
    asof = month_ends(data.calendar)[35]
    summary = run(data, tmp_path / "run", cfg, asof)
    assert summary["latest_selection_count"] == 20
    assert summary["data"]["synthetic"]
    audit = pd.read_csv(tmp_path / "run" / "training_audit.csv")
    trained = audit[audit.status == "trained"]
    assert (pd.to_datetime(trained.max_label_end) < pd.to_datetime(trained.cutoff)).all()
    nav = pd.read_csv(tmp_path / "run" / "daily_nav.csv")
    assert pd.to_datetime(nav.date).max() <= asof
    with pytest.raises(ValueError, match="not empty"):
        run(data, tmp_path / "run", cfg, asof)


@pytest.mark.parametrize("kwargs", [{"train_horizon": 1}, {"top_k": 0}, {"sell_tax_bps": -1},
                                    {"commission_bps": np.nan}, {"sell_tax_bps": np.inf},
                                    {"commission_bps": 10000}, {"top_k": 2.5}, {"embargo_sessions": 0.5}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_empty_prices_rejected(data):
    with pytest.raises(ValueError, match="must not be empty"):
        validate(replace(data, prices=data.prices.iloc[:0]))


def test_timezone_session_dates_rejected(data):
    p = data.prices.copy()
    p["date"] = p.date.dt.tz_localize("Asia/Shanghai")
    with pytest.raises(ValueError, match="timezone-free"):
        validate(replace(data, prices=p))


def test_bankrupt_nav_is_serializable():
    import json
    nav = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4), "nav": [1.0, 0.0, 0.0, 0.0]})
    metrics = nav_metrics(nav)
    assert metrics["total_return"] == -1
    assert metrics["max_drawdown"] == -1
    assert metrics["sharpe"] is None
    json.dumps(metrics, allow_nan=False)


def test_missing_benchmark_mean_serializes_null():
    assert _finite_mean(pd.Series([np.nan, np.nan])) is None
    assert _finite_mean(pd.Series([0.01, np.nan])) == 0.01


def test_cohort_costs_match_entry_and_exit_cashflows(data):
    cfg = Config(top_k=1, commission_bps=20, slippage_bps=30, sell_tax_bps=50)
    date = month_ends(data.calendar)[20]
    labels = forward_labels(data, [date])
    symbol = data.prices.symbol.iloc[0]
    selected = pd.DataFrame([{"date": date, "symbol": symbol}])
    result = cohort_performance(selected, labels, selected, data, cfg)
    buy_rate, sell_rate = 0.005, 0.01
    for row in result.itertuples():
        assert row.indicative_net_return == pytest.approx((1 + row.gross_return) / (1 + buy_rate) * (1 - sell_rate) - 1)


def test_asof_clips_data_before_generating_labels(data, tmp_path, monkeypatch):
    import ashare_mvp.pipeline as pipeline
    date = month_ends(data.calendar)[35]
    original = pipeline.forward_labels
    called = []

    def spy(observed, dates, horizons=(3, 6, 12)):
        assert observed.prices.date.max() <= date
        assert observed.calendar.max() <= date
        assert observed.fundamentals.available_date.max() <= date
        called.append(True)
        return original(observed, dates, horizons)

    monkeypatch.setattr(pipeline, "forward_labels", spy)
    result = run(data, tmp_path / "asof", Config(min_train_groups=4, n_estimators=3), date)
    assert called and result["latest_selection_count"] == 20


def test_invalid_asof_does_not_create_output(data, tmp_path):
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="observed exchange session"):
        run(data, output, asof="2000-01-01")
    assert not output.exists()
