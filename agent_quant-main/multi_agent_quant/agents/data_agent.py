from __future__ import annotations

import os
import time
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Tuple, List

import pandas as pd
import numpy as np

try:
    import akshare as ak
except Exception:
    ak = None


Adjust = Literal["", "qfq", "hfq"]
Period = Literal["daily", "weekly", "monthly"]


@dataclass
class DataAgentConfig:
    data_dir: str = "data/raw"
    cache_subdir: str = "akshare_cache"
    period: Period = "daily"
    adjust: Adjust = "qfq"

    # 稳定性：重试 + 退避 + 抖动 + 节流
    max_retries: int = 6
    base_backoff_sec: float = 1.2
    jitter_sec: float = 0.8
    polite_sleep_sec: float = 0.6
    chunk_sleep_sec: float = 1.2

    # 分段策略：按年拉取（更稳）
    enable_chunking: bool = True
    chunk_years: int = 1

    # 数据校验
    min_rows: int = 30

    # 缓存：parquet 优先，无引擎自动 fallback CSV
    prefer_parquet: bool = True


class DataAgent:
    """
    DataAgent（工业级最小稳定版）
    --------------------------------
    - 运行时输入 symbol/start/end 即可返回可回测 DataFrame
    - 数据源优先级（自动 fallback）：
        1) 东财：ak.stock_zh_a_hist（可能 RemoteDisconnected）
        2) 腾讯：ak.stock_zh_a_hist_tx（常作为备选）
        3) 新浪：ak.stock_zh_a_daily（你环境里可能 KeyError('date')，已捕获）
    - 禁用系统代理（避免 ProxyError），finally 恢复环境变量
    - 本地缓存：优先 parquet（若无 pyarrow/fastparquet 自动写 CSV）
    - 大区间支持分段拉取（按年/按两年）
    - 清洗标准化：DatetimeIndex + open/high/low/close/volume(+amount)
    """

    def __init__(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        adjust: str = "qfq",
        data_dir: str = "data/raw",
        cfg: Optional[DataAgentConfig] = None,
    ):
        self.symbol = str(symbol)
        self.start_date = str(start_date)  # YYYYMMDD
        self.end_date = str(end_date)      # YYYYMMDD
        self.adjust: Adjust = adjust if adjust in ("", "qfq", "hfq") else "qfq"

        self.cfg = cfg or DataAgentConfig()
        self.cfg.data_dir = data_dir
        self.cfg.adjust = self.adjust

        self.cache_root = Path(self.cfg.data_dir) / self.cfg.cache_subdir
        self.cache_root.mkdir(parents=True, exist_ok=True)

    # -----------------------
    # Public API
    # -----------------------
    def get_feature_data(
        self,
        use_cache: bool = True,
        force_refresh: bool = False,
        add_indicators: bool = True,
    ) -> pd.DataFrame:
        df = self.fetch_data(use_cache=use_cache, force_refresh=force_refresh)
        if add_indicators:
            df = self.add_indicators(df)
        return df

    def fetch_data(self, use_cache: bool = True, force_refresh: bool = False) -> pd.DataFrame:
        parquet_path, csv_path = self._cache_paths()

        # 1) 缓存命中直接读
        if use_cache and (not force_refresh):
            df = self._try_load_cache(parquet_path, csv_path)
            if df is not None:
                df = self._postprocess_loaded(df)
                self._validate_df(df)
                return df.loc[self._to_dt(self.start_date): self._to_dt(self.end_date)]

        # 2) 拉取（先整体，失败再分段）
        df_raw = self._fetch_from_akshare_stable()
        df = self._clean_data(df_raw)
        self._validate_df(df)

        # 3) 写缓存
        self._save_cache(df, parquet_path, csv_path)

        return df.loc[self._to_dt(self.start_date): self._to_dt(self.end_date)]

    # -----------------------
    # Cache IO
    # -----------------------
    def _try_load_cache(self, parquet_path: Path, csv_path: Path) -> Optional[pd.DataFrame]:
        if parquet_path.exists():
            try:
                return pd.read_parquet(parquet_path)
            except Exception:
                pass
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path, index_col=0, parse_dates=True)
            except Exception:
                return None
        return None

    def _save_cache(self, df: pd.DataFrame, parquet_path: Path, csv_path: Path) -> None:
        if self.cfg.prefer_parquet:
            try:
                df.to_parquet(parquet_path, index=True)
                return
            except ImportError:
                pass
            except Exception:
                pass
        df.to_csv(csv_path, index=True, encoding="utf-8-sig")

    def _cache_paths(self) -> Tuple[Path, Path]:
        key = f"{self.symbol}_{self.start_date}_{self.end_date}_{self.cfg.period}_{self.adjust}"
        h = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
        base = self.cache_root / f"{self.symbol}_{self.start_date}_{self.end_date}_{self.adjust}_{h}"
        return base.with_suffix(".parquet"), base.with_suffix(".csv")

    # -----------------------
    # Fetch (stable)
    # -----------------------
    def _fetch_from_akshare_stable(self) -> pd.DataFrame:
        if ak is None:
            raise RuntimeError("akshare is not available. Please `pip install akshare` first.")

        # 先整体拉取一次
        try:
            return self._fetch_once_with_retry(self.start_date, self.end_date)
        except Exception:
            if not self.cfg.enable_chunking:
                raise

        # 分段拉取
        chunks = self._split_date_ranges(self.start_date, self.end_date, years=self.cfg.chunk_years)
        parts: List[pd.DataFrame] = []
        for (s, e) in chunks:
            df_part = self._fetch_once_with_retry(s, e)
            parts.append(df_part)
            time.sleep(self.cfg.chunk_sleep_sec)

        df_all = pd.concat(parts, ignore_index=True)
        return df_all

    def _fetch_once_with_retry(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        单段拉取（多数据源 fallback，含禁代理/恢复环境）：
          A) 东财 stock_zh_a_hist（重试）
          B) 腾讯 stock_zh_a_hist_tx（重试）
          C) 新浪 stock_zh_a_daily（一次尝试+捕获 KeyError('date')）
        """
        last_err: Optional[Exception] = None

        proxy_keys = ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]
        old_env: Dict[str, Optional[str]] = {k: os.environ.get(k) for k in proxy_keys + ["NO_PROXY", "no_proxy"]}

        def _sleep_backoff(attempt: int) -> None:
            backoff = self.cfg.base_backoff_sec * (2 ** attempt)
            backoff += random.uniform(0.0, self.cfg.jitter_sec)
            time.sleep(backoff)

        def _tx_symbol(code: str) -> str:
            # 腾讯接口通常需要带市场前缀
            return ("sh" + code) if code.startswith("6") else ("sz" + code)

        try:
            # 禁用代理
            for k in proxy_keys:
                os.environ.pop(k, None)
            os.environ["NO_PROXY"] = "*"
            os.environ["no_proxy"] = "*"

            # ---------- A) 东财 ----------
            for attempt in range(self.cfg.max_retries):
                try:
                    time.sleep(self.cfg.polite_sleep_sec)
                    df = ak.stock_zh_a_hist(
                        symbol=self.symbol,
                        period=self.cfg.period,
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.adjust,
                    )
                    if df is None or len(df) == 0:
                        raise ValueError("eastmoney returned empty dataframe")
                    return df
                except Exception as e:
                    last_err = e
                    _sleep_backoff(attempt)

            # ---------- B) 腾讯（备选） ----------
            # 腾讯接口是日线历史，period 不同，因此只在 daily 场景下强推；非 daily 也尝试一次
            for attempt in range(max(2, self.cfg.max_retries // 2)):
                try:
                    time.sleep(self.cfg.polite_sleep_sec)
                    df_tx = ak.stock_zh_a_hist_tx(
                        symbol=_tx_symbol(self.symbol),
                        start_date=start_date,
                        end_date=end_date,
                        adjust=self.adjust,
                    )
                    if df_tx is None or len(df_tx) == 0:
                        raise ValueError("tencent returned empty dataframe")
                    return df_tx
                except Exception as e_tx:
                    last_err = e_tx
                    _sleep_backoff(attempt)

            # ---------- C) 新浪（最后兜底） ----------
            try:
                time.sleep(self.cfg.polite_sleep_sec)
                df_sina = ak.stock_zh_a_daily(
                    symbol=self.symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=self.adjust,
                )
                if df_sina is None or len(df_sina) == 0:
                    raise RuntimeError("sina returned empty dataframe")
                return df_sina
            except KeyError as e_key:
                # 你当前遇到的 KeyError('date') 就会落到这里
                last_err = e_key
            except Exception as e_sina:
                last_err = e_sina

            raise RuntimeError(f"All data sources failed for {self.symbol} {start_date}-{end_date}: {last_err}")

        finally:
            # 恢复原环境变量（必须保留）
            for k, v in old_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @staticmethod
    def _split_date_ranges(start: str, end: str, years: int = 1) -> List[Tuple[str, str]]:
        s = pd.to_datetime(start, format="%Y%m%d")
        e = pd.to_datetime(end, format="%Y%m%d")
        chunks: List[Tuple[str, str]] = []

        cur = s
        while cur <= e:
            year_end = pd.Timestamp(year=cur.year + years - 1, month=12, day=31)
            seg_end = min(year_end, e)
            chunks.append((cur.strftime("%Y%m%d"), seg_end.strftime("%Y%m%d")))
            cur = seg_end + pd.Timedelta(days=1)

        return chunks

    # -----------------------
    # Cleaning & Standardization
    # -----------------------
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # 兼容中文/英文/不同数据源字段
        rename_map = {
            # Chinese (eastmoney)
            "日期": "date",
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            # English (possible from other sources)
            "Date": "date",
            "date": "date",
            "Open": "open",
            "open": "open",
            "High": "high",
            "high": "high",
            "Low": "low",
            "low": "low",
            "Close": "close",
            "close": "close",
            "Volume": "volume",
            "volume": "volume",
            "Amount": "amount",
            "amount": "amount",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # index/日期处理
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.set_index("date")

        if not isinstance(df.index, pd.DatetimeIndex):
            # 某些接口可能 index 就是日期字符串/日期
            try:
                df.index = pd.to_datetime(df.index, errors="coerce")
            except Exception:
                pass

        df = df.sort_index()

        keep = [c for c in ["open", "high", "low", "close", "volume", "amount"] if c in df.columns]
        df = df[keep].copy()

        for c in keep:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=["close"])
        return df

    def _postprocess_loaded(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    # -----------------------
    # Indicators
    # -----------------------
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "close" not in df.columns:
            return df

        close = df["close"]

        for w in (5, 10, 20):
            col = f"ma_{w}"
            if col not in df.columns:
                df[col] = close.rolling(w).mean()

        if "return" not in df.columns:
            df["return"] = close.pct_change()

        if "vol_20" not in df.columns:
            df["vol_20"] = df["return"].rolling(20).std()

        if "rsi" not in df.columns:
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss
            df["rsi"] = 100 - (100 / (1 + rs))

        if "boll_upper" not in df.columns or "boll_lower" not in df.columns:
            m = close.rolling(20).mean()
            s = close.rolling(20).std()
            df["boll_upper"] = m + 2 * s
            df["boll_lower"] = m - 2 * s

        return df

    # -----------------------
    # Validation
    # -----------------------
    def _validate_df(self, df: pd.DataFrame) -> None:
        if df is None or len(df) == 0:
            raise ValueError("Data validation failed: empty dataframe")

        if len(df) < int(self.cfg.min_rows):
            raise ValueError(f"Data validation failed: too few rows ({len(df)} < {self.cfg.min_rows})")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Data validation failed: index is not DatetimeIndex")

        if not df.index.is_monotonic_increasing:
            raise ValueError("Data validation failed: index not sorted ascending")

        if "close" not in df.columns:
            raise ValueError("Data validation failed: missing 'close' column")

        if df["close"].isna().all():
            raise ValueError("Data validation failed: close all NaN")

        if (df["close"] <= 0).any():
            raise ValueError("Data validation failed: close contains non-positive values")

    @staticmethod
    def _to_dt(s: str) -> pd.Timestamp:
        return pd.to_datetime(s, format="%Y%m%d")