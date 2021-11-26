# 交易系统，封装回测、优化基本过程


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
def get_data(code, start_date = "20000101", end_date = "20201231", adjust = "qfq", period = "daily", refresh = False):
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
    if os.path.exists(stockfile) and refresh == False:
        stock_data = pd.read_csv(stockfile)
        stock_data.日期 = pd.to_datetime(stock_data.日期)
        stock_data.set_index("日期", drop = False, inplace = True)
    else:
        stock_data = download_data(code)
        if os.path.exists(stockfile):
            os.system("rm " + stockfile)
        stock_data.to_csv(stockfile)
    
    return stock_data
    
    
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
class Strategy(bt.Strategy):
    def __init__(self):
        pass
        
    def log(self, txt, dt = None):
        if self.params.bprint:
            dt = dt or self.datas[0].datetime.date(0)
            print('%s, %s' % (dt.isoformat(), txt))
                
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
            
            
# 回测类
class BackTest():
    """
        A股股票策略回测类
        strategy   回测策略
        code       回测股票代码
        start_date 回测开始日期
        end_date   回测结束日期
        stock_data 股票数据
        bk_data    基准数据
        rf         无风险收益率
        start_cash 初始资金
        stamp_duty 印花税率，单向征收
        commission 佣金费率，双向征收
        adjust     股票数据复权方式，qfq或hfq
        period     股票数据周期(日周月)
        refresh    是否更新数据
        bprint     是否输出中间结果
        bdraw      是否作图
        **param   策略参数，用于调参
    """
    def __init__(self, strategy, code, start_date, end_date, stock_data, bk_data, rf = 0.03, start_cash = 10000000, stamp_duty=0.005, commission=0.0001, adjust = "hfq", period = "daily", refresh = False, bprint = False, bdraw = False, **param):
        self._cerebro = bt.Cerebro()
        self._strategy = strategy
        self._code = code
        self._start_date = start_date
        self._end_date = end_date
        self._stock_data = stock_data
        self._bk_data = bk_data
        self._rf = rf
        self._start_cash = start_cash
        self._comminfo = CNA_Commission(stamp_duty=0.005, commission=0.0001)
        self._adjust = adjust
        self._period = period
        self._refresh = refresh
        self._bprint = bprint
        self._bdraw = bdraw
        self._param = param
        
    # 回测前准备
    def _before_test(self):
        self._data = self._datatransform(self._stock_data, self._code)
        self._cerebro.adddata(self._data)
        self._cerebro.addstrategy(self._strategy, bprint = self._bprint, **self._param)
        # print("测试", self._param, self._param.keys(), self._param.values(), self._param.items(), **self._param)
        # for k, v in self._param.items():
        #    print(k, v)
    
        self._cerebro.broker.setcash(self._start_cash)
        self._cerebro.broker.addcommissioninfo(self._comminfo)
        
    # 数据转换
    def _datatransform(self, stock_data, code):
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
        return data
    
    # 增加分析器
    def _add_analyzer(self):
        self._cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
        self._cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name = "TA")
        self._cerebro.addanalyzer(bt.analyzers.TimeReturn, _name = "TR")
        self._cerebro.addanalyzer(bt.analyzers.SQN, _name = "SQN")
        self._cerebro.addanalyzer(bt.analyzers.Returns, _name = "Returns")
        self._cerebro.addanalyzer(bt.analyzers.TimeDrawDown, _name = "TimeDrawDown")
        self._cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='SharpeRatio', timeframe=bt.TimeFrame.Days, annualize=True, riskfreerate=self._rf)
        self._cerebro.addanalyzer(bt.analyzers.SharpeRatio_A, _name='SharpeRatio_A')
        self._cerebro.addanalyzer(CostAnalyzer, _name="Cost")
        
    # 运行回测
    def run(self):
        self._before_test()
        self._add_analyzer()
        self._results = self._cerebro.run()
        return self._get_results()
        
    # 获取回测结果
    def _get_results(self):
        # 计算基准策略收益率
        bk_ret = self._bk_data.收盘.pct_change()
        bk_ret.fillna(0.0, inplace = True)
        # print(bk_ret)
    
        if self._bdraw:
            self._cerebro.plot(style = "candlestick")
            plt.savefig("./output/"+self._code+"_result.jpg")
    
        testresults = self._backtest_result(self._results, bk_ret, rf = self._rf)
        end_value = self._cerebro.broker.getvalue()
        pnl = end_value - self._start_cash

        testresults["初始资金"] = self._start_cash
        testresults["回测开始日期"] = bk_ret.index[0].date()
        testresults["回测结束日期"] = bk_ret.index[-1].date()
        testresults["期末净值"] = end_value
        testresults["净收益"] = pnl
        testresults["收益/成本"] = pnl/testresults["交易成本"]
        testresults["股票代码"] = self._code
        return testresults
        
    # 计算回测指标
    def _backtest_result(self, results, bk_ret, rf = 0.01):
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
        
        # print("测试", totalTrade)
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
        self._risk_analyze(backtest_results, returns, bk_ret, rf = rf)
    
        return backtest_results
        
    # 将风险分析和绘图部分提出来
    def _risk_analyze(self, backtest_results, returns, bk_ret, rf = 0.01):
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
        if self._bdraw:
            self._make_report(returns = returns, bk_ret = bk_ret, rf = rf)
        
    # 回测报告
    def _make_report(self, returns, bk_ret, rf, filename = "report.jpg", title = "回测结果", prepare_returns = False):
        quantstats.reports.html(returns = returns, benchmark = bk_ret, rf = rf, output='./output/stats.html', title=title, prepare_returns = prepare_returns)
        imgkit.from_file("./output/stats.html", "./output/" + filename, options = {"xvfb": ""})
        # 压缩图片文件
        im = Image.open("./output/" + filename)
        im.save("./output/" + filename)
        os.system("rm ./output/stats.html") 


# 实际的策略类
class MyStrategy(Strategy):
    params = (("maperiod", 15),
              ("bprint", False),)
    
    def __init__(self):
        super(MyStrategy, self).__init__()
        self.data_close = self.datas[0].close
        self.order = None
        self.buy_price = None
        self.buy_comm = None
        self.bbuy = False
        # 移动均线指标
        self.sma = bt.indicators.SimpleMovingAverage(self.datas[0], period = self.params.maperiod)
        
    def next(self):
        if self.order:
            return
            
        if not self.position:
            cash = self.broker.getcash()
            price = self.data_close[0]
            if price <= 0.0:
                return
            stake = math.ceil((0.95*cash/price)/100)*100
            if self.data_close[0] > self.sma[0]:
                self.order = self.buy(size = stake)
        else:
            if self.data_close[0] < self.sma[0]:
                self.order = self.close()
                
                
# 对整个市场的股票进行回测
class Research():
    """
        A股市场回测类
        strategy   回测策略
        start_date 回测开始日期
        end_date   回测结束日期
        highprice  筛选股票池的最高股价
        lowprice   筛选股票池的最低股价
        min_len    股票数据最小大小(避免新股等)
        start_cash 初始资金大小
        retest     是否重新回测
        refresh    是否更新数据
        bdraw      是否作图
    """
    def __init__(self, strategy, start_date, end_date, highprice = sys.float_info.max, lowprice = 0.0, min_len = 1, start_cash = 10000000, retest = False, refresh = False, bdraw = True):
        self._strategy = strategy
        self._start_date = start_date
        self._end_date = end_date
        self._highprice = highprice
        self._lowprice = lowprice
        self._min_len = min_len
        self._start_cash = start_cash
        self._retest = retest
        self._refresh = refresh
        self._bdraw = bdraw
        
    # 调用接口
    def run(self):
        self._test()
        if self._bdraw:
            self._draw(self._results)
        return self._results
        
    # 对回测结果画图
    def _draw(self, results):
        results.set_index("股票代码", inplace = True)
        # 绘图
        plt.figure()
        results.loc[:, ["SQN", "α值", "β值", "交易总次数", "信息比例", "夏普比率", "年化收益率", "收益/成本", "最大回撤", "索提比例", "胜率", "赔率"]].hist(bins = 100, figsize = (40, 20))
        plt.suptitle("对整个市场回测结果")
        plt.savefig("./output/market_test.jpg")
        
    # 执行回测    
    def _test(self):
        result_path = "./output/market_test.csv"
        if os.path.exists(result_path) and self._retest == False:
            self._results = pd.read_csv(result_path, dtype = {"股票代码":str})
            return
            
        self._codes = self._make_pool(refresh = self._refresh)
        self._results = pd.DataFrame()
        n = len(self._codes)
        i = 0
        bk_data = get_data(code = "000300", 
        start_date = self._start_date, 
        end_date = self._end_date,
        refresh = True)
        print("回测整个市场……")
        for code in self._codes:
            i += 1
            print("回测进度:", i/n)
            data = get_data(code = code, 
                start_date = self._start_date, 
                end_date = self._end_date,
                refresh = True)
            if len(data) <= self._min_len:
                continue
            backtest = BackTest(strategy = self._strategy, code = code, start_date = self._start_date, end_date = self._end_date, stock_data = data, bk_data = bk_data, start_cash = self._start_cash, refresh = True)
            res = backtest.run()
            self._results = self._results.append(res, ignore_index = True)
        self._results.to_csv(result_path)
        return
        
    # 形成股票池
    def _make_pool(self, refresh = True):
        data = pd.DataFrame()
        path = "./datas/"
        stockfile = path + "stocks.csv"
        if os.path.exists(stockfile) and refresh == False:
            data = pd.read_csv(stockfile, dtype = {"code":str, "昨日收盘":np.float64})
        else:
            stock_zh_a_spot_df = ak.stock_zh_a_spot()
            stock_zh_a_spot_df.to_csv(stockfile)
            data = stock_zh_a_spot_df
        codes = self._select(data)
        return codes
        
    # 对股票数据进行筛选
    def _select(self, data, highprice = sys.float_info.max, lowprice = 0.0):
        # 对股价进行筛选
        smalldata = data[(data.最高 < highprice) & (data.最低 > lowprice)]
        # 排除ST个股
        smalldata = smalldata[~ smalldata.名称.str.contains("ST")]
        # 排除要退市个股
        smalldata = smalldata[~ smalldata.名称.str.contains("退")]

        codes = []
        for code in smalldata.代码.values:
            codes.append(code[2:])
    
        return codes

       
@run.change_dir
def main():
    init_display()
    data = get_data(code = "513100", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    bk_data = get_data(code = "000300", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    backtest = BackTest(
        strategy = MyStrategy, 
        code = "513100", 
        start_date = "20160101", 
        end_date = "20211231", 
        stock_data = data, 
        bk_data = bk_data,
        rf = 0.03, 
        start_cash = 10000000,
        stamp_duty=0.005, 
        commission=0.0001, 
        adjust = "hfq", 
        period = "daily", 
        refresh = True, 
        bprint = False, 
        bdraw = True)
    results = backtest.run()
    print("回测结果", results)
    

# 对整个市场进行回测
@run.change_dir
def do_research():
    init_display()
    research = Research(MyStrategy, start_date = "20150101", end_date = "20210101", min_len = 100, retest = False)
    results = research.run()
    print(results.head(), results.describe())
    
    
# 对策略进行参数优化
@run.change_dir
def optimize():
    init_display()
    data = get_data(code = "513100", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    bk_data = get_data(code = "000300", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    map = range(2, 20)
    opt_results = pd.DataFrame()
    params = []
    for i in map:
        backtest = BackTest(
                strategy = MyStrategy, 
                code = "513100", 
                start_date = "20160101", 
                end_date = "20211231", 
                stock_data = data, 
                bk_data = bk_data,
                rf = 0.03, 
                start_cash = 10000000,
                stamp_duty=0.005, 
                commission=0.0001, 
                adjust = "hfq", 
                period = "daily", 
                refresh = False, 
                bprint = False, 
                bdraw = False,
                maperiod = i)
        results = backtest.run()
        opt_results = opt_results.append(results, ignore_index = True)
        params.append(i)
    opt_results["参数"] = params
    # 按年化收益率排序
    opt_results.sort_values(by = "年化收益率", inplace = True, ascending = False)
    print(opt_results.loc[:, ["参数", "年化收益率"]])


if __name__ == "__main__":
    # main()
    # do_research()
    optimize()
    # testA(a = 2, b = 5.6)