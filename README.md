# agent_quant_2 — A股多因子选股 MVP

新增 `ashare_mvp/` 为独立研究模块，旧 `agent_quant-main/` 演示保留。代码检查与使用说明见 [MVP说明](docs/MVP.md)，运行结果见 [开发与验证报告](docs/IMPLEMENTATION_REPORT.md)，复查修复与测试见 [Debug报告](docs/DEBUG_REPORT.md)。

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -e '.[test]'
.venv/Scripts/python -m pytest -q
.venv/Scripts/python -m ashare_mvp.cli demo --out artifacts/demo_data
.venv/Scripts/python -m ashare_mvp.cli run --data artifacts/demo_data --out artifacts/demo_run
```

示例输入为**合成数据**，仅用于检验训练/回测工程。真实选股需要提供具备公告版本、历史股票池、退市和停牌记录的数据。
