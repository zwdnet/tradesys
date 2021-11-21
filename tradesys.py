# AkShare上的例子


import backtrader as bt
import quantstats
import akshare as ak
import efinance as ef
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import run
import sys
import math
import imgkit
from PIL import Image
from scipy import stats
import empyrical as ey



# 设置显示环境
def init_display():
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    

# 获取数据
@run.change_dir
def get_data(code, bk_code = "000300", start_date = "20000101", end_date = "20201231", adjust = "qfq", period = "daily", refresh = False):
    def download_data(code):
        try:
            data = ak.stock_zh_a_hist(symbol = code, start_date = start_date, end_date = end_date, adjust = adjust, period = period)
        except KeyError:
            if adjust == "qfq":
                fqt = 1
            elif adjust == "hfq":
                fqt = 2
            
            if period == "daily":
                klt = 101
            elif period == "weekly":
                klt = 102
            elif period == "monthly":
                klt = 103
            data = ef.stock.get_quote_history(code, beg = start_date, end = end_date, fqt = fqt, klt = klt)
        data.日期 = pd.to_datetime(data.日期)
        data.set_index("日期", drop = False, inplace = True)
        return data
            
    stockfile = "./datas/"+code+".csv"
    bkfile = "./datas/"+bk_code+".csv"
    if os.path.exists(stockfile) and refresh == False:
        stock_data = pd.read_csv(stockfile)
        stock_data.日期 = pd.to_datetime(stock_data.日期)
        stock_data.set_index("日期", drop = False, inplace = True)
    else:
        stock_data = download_data(code)
        stock_data.to_csv(stockfile)
    
    # 获取基准数据
    if os.path.exists(bkfile) and refresh == False:
        bk_data = pd.read_csv(bkfile)
        bk_data.日期 = pd.to_datetime(bk_data.日期)
        bk_data.set_index("日期", drop = False, inplace = True)
    else:
        bk_data = download_data(bk_code)
        bk_data.to_csv(bkfile)
    
    # 生成datafeed
    data = bt.feeds.PandasData(
            dataname=stock_data,
            name=code,
            fromdate=stock_data.日期[0],
            todate=stock_data.日期[len(stock_data) - 1],
            datetime='日期',
            open='开盘',
            high='最高',
            low='最低',    
            close='收盘',
            volume='成交量',
            openinterest=-1
        )
    return (data, bk_data)
    
    
# A股的交易成本:买入交佣金，卖出交佣金和印花税
class CNA_Commission(bt.CommInfoBase):
    params = (('stamp_duty', 0.005), # 印花税率 
              ('commission', 0.0001), # 佣金率 
              ('stocklike', True),   ('commtype', bt.CommInfoBase.COMM_PERC),)
    
    def _getcommission(self, size, price, pseudoexec):
        if size > 0:
            return size * price * self.p.commission
        elif size < 0:
            return - size * price * (self.p.stamp_duty + self.p.commission)
        else:
            return 0
            
            
# 自定义分析器，记录交易成本数据
class CostAnalyzer(bt.Analyzer):
    def __init__(self):
        self._cost = []
        self.ret = 0.0
        
    def notify_trade(self, trade):
        if trade.justopened or trade.status == trade.Closed:
            self._cost.append(trade.commission)
            
    def stop(self):
        super(CostAnalyzer, self).stop()
        self.ret = np.sum(self._cost)
        
    def get_analysis(self):
        return self.ret

    
# 均线策略
class MyStrategy(bt.Strategy):
    params = (("maperiod", 15),
              ("bprint", False),)
    
    def __init__(self):
        self.data_close = self.datas[0].close
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.bbuy = False
        # 移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.params.maperiod)
        
    def log(self, txt, dt = None):
        if self.params.bprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            cash = self.broker.getcash()
            price = self.data_close[0]
            stake = math.ceil((0.95*cash/price)/100)*100
            # self.log(f"{cash} {price} {stake} {stake*price}")
            if self.data_close[0] > self.sma[0]:
                self.order = self.buy(size = stake)
        else:
            if self.data_close[0] < self.sma[0]:
                self.order = self.close()
        """
        if self.bbuy == False:
            self.order = self.buy(size = 900000)
        """
                
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log("交易被拒绝/现金不足/取消")
        elif order.status in [order.Completed]: 
            if order.isbuy(): 
                self.log('买单执行,%s, %.2f, %i' % (order.data._name, order.executed.price, order.executed.size))
            elif order.issell(): 
                self.log('卖单执行, %s, %.2f, %i' % (order.data._name, order.executed.price, order.executed.size))
        self.order = None
        
    def notify_trade(self, trade): 
        if trade.isclosed: 
            self.log('毛收益 %0.2f, 扣佣后收益 % 0.2f, 佣金 %.2f, 市值 %.2f, 现金 %.2f'%(trade.pnl, trade.pnlcomm, trade.commission, self.broker.getvalue(), self.broker.getcash()))
                
    def stop(self):
        if self.position:
            self.close()
            
            
# 计算回测指标
@run.change_dir
def backtest_result(results, bk_ret, rf = 0.01):
    # 计算回测指标
    portfolio_stats = results[0].analyzers.getbyname('PyFolio')
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    returns.index = returns.index.tz_convert(None)
    totalTrade = results[0].analyzers.getbyname("TA").get_analysis()
    sqn = results[0].analyzers.SQN.get_analysis()["sqn"]
    Returns = results[0].analyzers.Returns.get_analysis()
    timedrawdown = results[0].analyzers.TimeDrawDown.get_analysis()
    sharpe = results[0].analyzers.SharpeRatio.get_analysis()
    sharpeA = results[0].analyzers.SharpeRatio_A.get_analysis()
    cost = results[0].analyzers.Cost.get_analysis()
    backtest_results = pd.Series()
    # print("每日收益率序列", returns)
    # backtest_results["收益率序列"] = returns
    # backtest_results["夏普比率bt"] = sharpe
    # backtest_results["年化夏普比率bt"] = sharpeA
    backtest_results["总收益率"] = Returns["rtot"]
    backtest_results["平均收益率"] = Returns["ravg"]
    backtest_results["年化收益率"] = Returns["rnorm"]
    backtest_results["交易成本"] = cost
    backtest_results["交易总次数"] = totalTrade["total"]["total"]
    backtest_results["盈利交易次数"] = totalTrade["won"]["total"]
    backtest_results["盈利交易总盈利"] = totalTrade["won"]["pnl"]["total"]
    backtest_results["盈利交易平均盈利"] = totalTrade["won"]["pnl"]["average"]
    backtest_results["盈利交易最大盈利"] = totalTrade["won"]["pnl"]["max"]
    backtest_results["亏损交易次数"] = totalTrade["lost"]["total"]
    backtest_results["亏损交易总亏损"] = totalTrade["lost"]["pnl"]["total"]
    backtest_results["亏损交易平均亏损"] = totalTrade["lost"]["pnl"]["average"]
    backtest_results["亏损交易最大亏损"] = totalTrade["lost"]["pnl"]["max"]
    backtest_results["SQN"] = sqn
    # 胜率就是成功率，例如投入十次，七次盈利，三次亏损，胜率就是70%。
    backtest_results["胜率"] = totalTrade["won"]["total"]/totalTrade["total"]["total"]
    # 赔率是指盈亏比，例如平均每次盈利30%，平均每次亏损10%，盈亏比就是3倍。
    backtest_results["赔率"] = totalTrade["won"]["pnl"]["average"]/abs(totalTrade["lost"]["pnl"]["average"])
    
    # 计算风险指标
    risk_analyze(backtest_results, returns, bk_ret, rf = rf)
    
    return backtest_results
    
    
# 将风险分析和绘图部分提出来，要debug的
@run.change_dir
def risk_analyze(backtest_results, returns, bk_ret, rf = 0.01):
    print("测试", returns.tail(), bk_ret.tail())
    prepare_returns = False # 已经是收益率序列数据了，不用再转换了
    # 计算夏普比率
    if returns.std() == 0.0:
        sharpe = 0.0
    else:
        sharpe = quantstats.stats.sharpe(returns = returns, rf = rf)
    # 计算αβ值
    alphabeta = quantstats.stats.greeks(returns, bk_ret, prepare_returns = prepare_returns)
    # 计算信息比率
    info = quantstats.stats.information_ratio(returns, bk_ret, prepare_returns = prepare_returns)
    # 索提比率
    sortino = quantstats.stats.sortino(returns = returns, rf = rf)
    # 调整索提比率
    adjust_st = quantstats.stats.adjusted_sortino(returns = returns, rf = rf)
    # skew值
    skew = quantstats.stats.skew(returns = returns, prepare_returns = prepare_returns)
    # calmar值
    calmar = quantstats.stats.calmar(returns = returns, prepare_returns = prepare_returns)
    
    # r2值
    r2 = quantstats.stats.r_squared(returns, bk_ret, prepare_returns = prepare_returns)
    
    backtest_results["波动率"] = quantstats.stats.volatility(returns = returns, prepare_returns = prepare_returns)
    backtest_results["赢钱概率"] = quantstats.stats.win_rate(returns = returns, prepare_returns = prepare_returns)
    backtest_results["收益风险比"] = quantstats.stats.risk_return_ratio(returns = returns, prepare_returns = prepare_returns)
    
    backtest_results["夏普比率"] = sharpe
    backtest_results["α值"] = alphabeta.alpha
    backtest_results["β值"] = alphabeta.beta
    backtest_results["信息比例"] = info
    backtest_results["索提比例"] = sortino
    backtest_results["调整索提比例"] = adjust_st
    backtest_results["skew值"] = skew
    backtest_results["calmar值"] = calmar
    backtest_results["r2值"] = r2
    
    # 最大回撤
    md = quantstats.stats.max_drawdown(prices = returns)
    backtest_results["最大回撤"] = md
    
    # 生成回测报告
    make_report(returns = returns, bk_ret = bk_ret, rf = rf)
    
    
# 回测报告
@run.change_dir
def make_report(returns, bk_ret, rf, filename = "report.jpg", title = "回测结果", prepare_returns = False):
    quantstats.reports.html(returns = returns, benchmark = bk_ret, rf = rf, output='./output/stats.html', title=title, prepare_returns = prepare_returns)
    imgkit.from_file("./output/stats.html", "./output/" + filename, options = {"xvfb": ""})
    # 压缩图片文件
    im = Image.open("./output/" + filename)
    im.save("./output/" + filename)
    os.system("rm ./output/stats.html") 
    
    
# 初始化
def init():
    cerebro = bt.Cerebro()
    code = "513100"
    start_date = "20160101"
    end_date = "20211231"
    rf = 0.03
    start_cash = 10000000
    comminfo = CNA_Commission(stamp_duty=0.005, commission=0.0001)
    return cerebro, code, start_date, end_date, rf, start_cash, comminfo
    
    
# 回测前的处理
def before_test(cerebro, code, start_date, end_date, rf, start_cash, comminfo):
    data, bk_data = get_data(code = code, start_date = start_date, end_date = end_date, adjust = "hfq", refresh = False)
    cerebro.adddata(data)
    cerebro.addstrategy(MyStrategy, bprint = False)
    
    cerebro.broker.setcash(start_cash)
    
    cerebro.broker.addcommissioninfo(comminfo)
    return bk_data 
    
    
# 增加分析器
def add_analyzer(cerebro, rf):
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = "TA")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name = "TR")
    cerebro.addanalyzer(bt.analyzers.SQN, _name = "SQN")
    cerebro.addanalyzer(bt.analyzers.Returns, _name = "Returns")
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name = "TimeDrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=rf)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='SharpeRatio_A')
    cerebro.addanalyzer(CostAnalyzer, _name="Cost")   
    
    
# 获取回测数据
@run.change_dir
def get_results(cerebro, bk_data, rf, code, results, start_cash):
    # 计算基准策略收益率
    bk_ret = bk_data.收盘.pct_change()
    bk_ret.fillna(0.0, inplace = True)
    # print(bk_ret)
    
    cerebro.plot(style = "candlestick")
    plt.savefig("./output/"+code+"_result.jpg")
    
    testresults = backtest_result(results, bk_ret, rf = rf)
    end_value = cerebro.broker.getvalue()
    pnl = end_value - start_cash

    testresults["初始资金"] = start_cash
    testresults["回测开始日期"] = bk_ret.index[0].date()
    testresults["回测结束日期"] = bk_ret.index[-1].date()
    testresults["期末净值"] = end_value
    testresults["净收益"] = pnl
    testresults["收益/成本"] = pnl/testresults["交易成本"]
    return testresults
    

@run.change_dir                
def main():
    init_display()
    cerebro, code, start_date, end_date, rf, start_cash, comminfo = init()
    
    bk_data = before_test(cerebro, code, start_date, end_date, rf, start_cash, comminfo)
    
    # 增加分析器
    add_analyzer(cerebro, rf)
    results = cerebro.run()

    # 获得回测结果
    results = get_results(cerebro, bk_data, rf, code, results, start_cash)
    print(results)


if __name__ == "__main__":
    main()
    