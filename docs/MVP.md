# A股多因子选股 MVP

## 架构与用途

该模块是月度、中长期横截面选股研究工具。历史单标的策略演示不作为选股训练数据，也不用于证明该模块收益。

流程：数据契约校验 → 仅历史行情的技术指标 → 财报时点快照 → 横截面因子 → 3/6/12个月标签 → 按月滚动 LightGBM LambdaRank → Top20 → 月度调仓净值与持有期分组绩效。

`data.py` 管理数据契约和公告版本；`factors.py` 管理可扩展因子；`model.py` 管理训练和配置；`pipeline.py` 管理标签、滚动排序与回测；`cli.py` 提供本地入口。LightGBM 每个决策月是一组 query，标签为该组未来收益的五级相关性；预测分数用于排序，不是上涨概率。接口依据 [LightGBM 官方 LGBMRanker 文档](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMRanker.html)。

## 因子

默认10项：ROE、收入增长、利润增长、低负债率、盈利收益率、账面权益收益率、63/126交易日动量、252日动量跳过最近21日、63日低波动。

`--oneil` 加入3项可选近似指标：252日高点接近度、成交量扩张、利润增长减收入增长。它们只是可扩展特征，并非完整 CAN SLIM 规则，也不单独决定买卖。

扩展调用 `FactorLibrary.register(name, family, compute)`；计算函数只能接收当前历史快照。新增因子必须补充未来扰动不改变历史值的测试。输入基本面比率使用小数格式，例如 ROE 12% 为 `0.12`；财务金额与市值必须同币种同单位。

## 输入文件契约

目录需要四个文件，CSV 的 symbol 必须按字符串读取，保留前导零：

| 文件 | 必需字段 |
|---|---|
| prices.csv | date, symbol, close, total_return_open, volume, market_cap, is_st, eligible, can_buy, can_sell, listing_date |
| fundamentals.csv | symbol, report_date, available_date, roe, earnings_ttm, book_equity, revenue_growth, earnings_growth, debt_ratio |
| calendar.csv | date |
| metadata.json | source, synthetic, price_basis, universe_policy, availability_policy |

`date` 为上海交易日日期，不能带时分秒。`calendar.csv` 必须来自交易所交易日历；合成数据仅使用模拟工作日日历。四个布尔状态字段为0/1。`market_cap` 使用当时已知总市值，不能使用今天的市值填历史；`eligible` 与 `is_st` 为历史状态，不得用今天的上市股票清单回填整个历史。

`close` 与 `total_return_open` 必须采用一致的总回报调整基准，以正确处理分红和拆股；provider 需在 metadata 说明具体处理方法。市场价值单独提供，不由调整价格乘当前股本计算。要求 `price_basis="total_return_consistent"`，这项声明并不能替代供应商数据审计。

停牌需要提供估值且 `can_buy=can_sell=0`。数据覆盖从上市日起到日历末尾每个交易日；退市证券也必须有终止估值及后续延续记录，`eligible=0`，可为0估值，不能直接删除证券以躲避损失。模拟委托状态由实际执行日可买/可卖标志决定；不得通过未来交易状态预先删除股票。

`available_date` 是该版本最早可以在该日收盘决策时使用的日期。只有公告日期没有时间时，保守设为公告后的第一个交易日；盘后公告也转为下一交易日。修订财报保留每个版本，不覆盖旧记录。首先筛选 available_date <= 决策日，再取最新报告期的最新可见版本。`earnings_ttm`、ROE、增长率等同样须由当时可见数据计算，不能直接采用供应商最终修订序列。

示例 metadata：

```json
{
  "source": "供应商、版本及取得日期",
  "synthetic": false,
  "price_basis": "total_return_consistent",
  "universe_policy": "逐日历史上市/ST/退市状态，包含退市估值",
  "availability_policy": "公告时间转上海可用交易日，逐版本保留"
}
```

## 防泄漏与训练

1. 上市至少365日，排除当时ST、不适格、零成交量、非正市值；财报报告期距今不超过550日，关键字段和技术指标必须完整。
2. 因子按当日横截面1%/99%缩尾并标准化，不进行全样本拟合。财报时点快照和技术滚动指标不读未来数据。
3. 月末收盘信号，下一交易日总回报调整开盘价作为标签起点。终点为起点加3/6/12个自然月后第一个交易日；没有完整终点则收益为空，不缩短持有期。
4. 默认训练6个月收益排序，可用 `--horizon 3` 或 `--horizon 12`。默认60个月滚动窗口，至少12个成熟月份，每个训练月份至少20个合格样本。
5. 每次训练按实际 label_end 清除未成熟标签，并留一个交易日 embargo：label_end 必须严格早于截止日。每个月重新训练，无随机时间拆分、无测试期调参或全样本早停。
6. `--asof` 决定研究截止日，可输出该交易日最新排名；历史评价只使用截至该日已成熟的结果。日历末月不自动假定是完整月，另输出最新排序。

## 执行与绩效解释

历史每月 Top20 等权目标仓位，次日开盘模拟调仓。先卖再买，交易费和滑点扣除，买入资金不足时按比例缩小。卖出受阻保留原持仓；买入受阻保留现金，没有未来收益驱动的补位。日净值在开盘估值，最终不强制清算。3/6/12个月绩效为重叠的持有期组合，不是月度策略净值，也不能连乘或视为独立样本。

每个持有期组合必须全部20只标签成熟才给出组合收益，避免只平均仍有价格的股票。对照为当时因子有效股票池等权持有期收益，**不是沪深300或全A指数**。输出 gross_return、indicative_net_return、universe_return、excess_return，并附执行端点是否可交易。持有期净收益使用 `(1 + 毛收益) × (1 - 卖出费率) / (1 + 买入费率) - 1`，按统一费率现金流计算；不可执行端点仍保留经济收益并标记，不能当作实盘可实现收益。月度净值包含实际模拟受阻订单和交易成本。

默认佣金3bps/侧、滑点5bps/侧、卖出税5bps为研究假设，可通过参数覆盖，未实现历史税率变化。当前使用调整价格与碎股模拟，没有100股整手、最低佣金、原始股价现金分红/除权记账和完整T+1订单系统；因此是研究MVP，尚非实盘交易引擎。

## 运行与输出

```powershell
python -m ashare_mvp.cli run --data path/to/pit_data --out artifacts/real_run --horizon 6 --oneil
python -m ashare_mvp.cli run --data path/to/pit_data --out artifacts/asof_run --asof 2024-06-28
```

输出目录必须为空，防止把不同实验混在一起。输出包含月度及最新 Top20、训练时点审计、因子重要性、逐日净值、委托、3/6/12个月持有期绩效、JSON摘要及每期模型文件。JSON保留配置、输入哈希和数据来源声明。历史截止日运行在生成因子、标签和模型前先截断行情及财报，哈希也仅覆盖截止日前输入。交易日历仍可用于判断已结束月份，但不读取未来行情结果。缺失基准均值输出JSON null，全损后无定义的Sharpe也输出null。模型分数跨月不可直接比较；gain 重要性只说明模型用到哪些特征，不构成因果解释。

真实数据接口已经可用，但没有自动下载财报历史版本。首次上线应先对供应商数据进行样本对账，再引入行业中性化、历史指数基准、独立验证窗口、组合约束与原始股价订单引擎。
