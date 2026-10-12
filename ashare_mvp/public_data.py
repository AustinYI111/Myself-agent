"""Public-source staging: raw provenance, never auto-certify PIT revisions."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import socket

import pandas as pd


def _result_frame(result):
    if result.error_code != "0":
        raise RuntimeError(f"provider {result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"provider pagination {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def is_ashare(code):
    return bool(re.fullmatch(r"(?:sh\.(?:600|601|603|605|688)\d{3}|sz\.(?:000|001|002|003|300|301)\d{3})", code))


def download_public(out, start, end, symbols=None, max_symbols=5, timeout=15, client=None):
    """Download raw samples and dated universe snapshots; explicit bounded run.

    max_symbols=None requests the full historical union. BaoStock coverage is
    Shanghai/Shenzhen; Beijing coverage is not invented or silently asserted.
    """
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if start > end or start != start.normalize() or end != end.normalize():
        raise ValueError("start/end must be ordered session dates")
    if timeout <= 0 or (max_symbols is not None and max_symbols < 1):
        raise ValueError("invalid timeout/max_symbols")
    path = Path(out)
    if path.exists() and any(path.iterdir()):
        raise ValueError("public staging output must be empty")
    if client is None:
        import baostock as client
    path.mkdir(parents=True, exist_ok=True)
    manifest = dict(source="BaoStock", retrieved_at=datetime.now(timezone.utc).isoformat(),
                    start=str(start.date()), end=str(end.date()), synthetic=False, files={}, queries=[],
                    requested_symbols=symbols, max_symbols=max_symbols,
                    training_ready=False,
                    blockers=["Financial original/revision vintage not certified",
                              "No audited limit-up/down execution flags or terminal delisting valuations",
                              "Adjusted price total-return methodology requires reconciliation",
                              "Public coverage does not prove all A shares including Beijing"])
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    logged_in = False

    def fetch(name, query):
        try:
            frame = _result_frame(query())
            file = path / f"{name}.csv"
            frame.to_csv(file, index=False)
            manifest["files"][file.name] = dict(rows=len(frame), columns=list(frame.columns),
                                                sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            manifest["queries"].append(dict(name=name, status="ok", rows=len(frame)))
            return frame
        except Exception as exc:
            manifest["queries"].append(dict(name=name, status="failed", error=str(exc)))
            return None

    try:
        login = client.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login {login.error_code}: {login.error_msg}")
        logged_in = True
        calendar = fetch("trade_dates", lambda: client.query_trade_dates(str(start.date()), str(end.date())))
        if calendar is None or calendar.empty:
            raise RuntimeError("no exchange calendar returned")
        sessions = pd.DatetimeIndex(pd.to_datetime(calendar.loc[calendar.is_trading_day == "1", "calendar_date"]))
        if sessions.empty:
            raise ValueError("no sessions in requested interval")
        # Monthly historical union; daily ST/status comes from historical bars.
        checkpoints = pd.Series(sessions).groupby(sessions.to_period("M")).last().tolist()
        checkpoints = sorted(set([sessions[0], *checkpoints]))
        universe = set()
        for day in checkpoints:
            frame = fetch(f"universe_{day:%Y%m%d}", lambda d=day: client.query_all_stock(str(d.date())))
            if frame is not None:
                universe.update(c for c in frame.code if is_ashare(c))
        chosen = list(symbols) if symbols is not None else sorted(universe)
        if not all(is_ashare(c) for c in chosen) or len(chosen) != len(set(chosen)):
            raise ValueError("symbols must be unique Shanghai/Shenzhen A-share codes, e.g. sh.600000")
        if max_symbols is not None:
            chosen = chosen[:max_symbols]
        manifest["downloaded_symbols"] = chosen
        fields = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST,peTTM,pbMRQ"
        for code in chosen:
            key = code.replace(".", "_")
            fetch(f"basic_{key}", lambda c=code: client.query_stock_basic(code=c))
            for adjust, label in [("3", "raw"), ("1", "adjusted")]:
                fetch(f"prices_{label}_{key}", lambda c=code, a=adjust: client.query_history_k_data_plus(
                    c, fields, start_date=str(start.date()), end_date=str(end.date()), frequency="d", adjustflag=a))
            for period in pd.period_range(start - pd.DateOffset(years=1), end, freq="Q"):
                for kind in ["profit", "growth", "balance"]:
                    fetch(f"{kind}_{key}_{period.year}Q{period.quarter}",
                          lambda c=code, p=period, k=kind: getattr(client, f"query_{k}_data")(
                              code=c, year=p.year, quarter=p.quarter))
        fetch("industry_current", client.query_stock_industry)
        manifest["status"] = "partial" if any(q["status"] == "failed" for q in manifest["queries"]) else "downloaded_raw"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
    finally:
        if logged_in:
            try:
                client.logout()
            except Exception:
                pass
        socket.setdefaulttimeout(old_timeout)
        (path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
