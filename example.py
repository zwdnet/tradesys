# 使用交易系统代码进行回测的例子


import tradesys as ts
import backtrader as bt
import run
import math


# 实际的策略类
class MyStrategy(ts.Strategy):
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


# 对单只股票进行回测
@run.change_dir
def back_test():
    ts.init_display()
    data = ts.get_data(code = "513100", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    bk_data = ts.get_data(code = "000300", 
        start_date = "20160101", 
        end_date = "20211231",
        refresh = True)
    backtest = ts.BackTest(
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
    ts.init_display()
    research = ts.Research(MyStrategy, start_date = "20150101", end_date = "20210101", min_len = 100, retest = False, refresh = False, maperiod = 19)
    results = research.run()
    print(results.head(), results.describe())
    print(results.isnull().any())
    
    
# 进行调参
@run.change_dir    
def do_opt():
    ts.init_display()
    map = range(2, 20)
    opt = ts.OptStrategy(
        code = "513100", 
        bk_code = "000300", 
        strategy = MyStrategy, 
        start_date = "20160101", 
        end_date = "20211231", 
        start_cash = 10000000, 
        retest = False, 
        refresh = False, 
        bprint = False,
        bdraw = False, 
        maperiod = range(10, 20))
    results = opt.run()
    results = opt.sort_results(results, key = "年化收益率")
    print(results.loc[:, ["参数", "年化收益率"]])


if __name__ == "__main__":
    # back_test()
    do_research()
    # do_opt()
    