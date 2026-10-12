import argparse
import json

from .data import load
from .demo import make_demo, save
from .model import Config
from .pipeline import run


def main():
    parser = argparse.ArgumentParser(description="A-share monthly factor ranker research MVP")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="generate explicitly synthetic input data")
    demo.add_argument("--out", required=True)
    demo.add_argument("--stocks", type=int, default=40)
    demo.add_argument("--sessions", type=int, default=1512)
    demo.add_argument("--seed", type=int, default=42)
    public = sub.add_parser("download-public", help="download raw public data; does not certify PIT")
    public.add_argument("--out", required=True)
    public.add_argument("--start", required=True)
    public.add_argument("--end", required=True)
    public.add_argument("--symbols", nargs="+")
    public.add_argument("--max-symbols", type=int, default=5)
    public.add_argument("--all-symbols", action="store_true")
    backtest = sub.add_parser("run", help="walk-forward training, rankings and performance")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--out", required=True)
    backtest.add_argument("--asof")
    backtest.add_argument("--horizon", type=int, choices=[3, 6, 12], default=6)
    backtest.add_argument("--oneil", action="store_true")
    backtest.add_argument("--top-k", type=int, default=20)
    backtest.add_argument("--train-months", type=int, default=60)
    backtest.add_argument("--min-train-groups", type=int, default=12)
    backtest.add_argument("--commission-bps", type=float, default=3)
    backtest.add_argument("--slippage-bps", type=float, default=5)
    backtest.add_argument("--sell-tax-bps", type=float, default=5)
    args = parser.parse_args()
    if args.command == "demo":
        save(make_demo(args.seed, args.stocks, args.sessions), args.out)
        print(f"SYNTHETIC data saved to {args.out}")
    elif args.command == "download-public":
        from .public_data import download_public
        result = download_public(args.out, args.start, args.end, args.symbols,
                                 None if args.all_symbols else args.max_symbols)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] in ["failed", "partial"]:
            raise SystemExit(1)
    else:
        config = Config(train_horizon=args.horizon, oneil=args.oneil, top_k=args.top_k,
                        train_months=args.train_months, min_train_groups=args.min_train_groups,
                        commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
                        sell_tax_bps=args.sell_tax_bps)
        print(json.dumps(run(load(args.data), args.out, config, args.asof), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
