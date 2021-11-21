# 测试程序


from tradesys import *


# 测试风险指标的计算
def test_risk():
    # 创建测试数据
    def make_data():
        date = pd.date_range("1/1/2021", "1/10/2021")
        priceA = [1.0, 1.1, 1.2, 1.3, 0.5, 0.8, 0.4, 0.6, 1.0, 1.2]
        priceB = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09]
    
        print(priceA, priceB)
        data = pd.DataFrame({"日期": date,
                         "开盘": priceA,
                         "收盘": priceA,
                         "最高": priceA,
                         "最低": priceA,
                         "成交量": priceA
        })
    
        bench = pd.DataFrame({"日期": date,
                         "开盘": priceB,
                         "收盘": priceB,
                         "最高": priceB,
                         "最低": priceB,
                         "成交量": priceB
        })
        data.set_index("日期", drop = True, inplace = True)
        bench.set_index("日期", drop = True, inplace = True)

        return (data, bench)

    days = 252
    rf = 0.01
    print("测试风险指标计算")
    data, bench = make_data()
    data_ret = data.收盘.pct_change().fillna(0).tz_localize(None)
    bench_ret = bench.收盘.pct_change().fillna(0).tz_localize(None)
    print(data_ret, bench_ret)
    test_results = pd.Series()
    # print(data.策略收益率, data.基准收益率)
    risk_analyze(test_results, data_ret, bench_ret, rf = rf)
    print(test_results)
    
    print("手算")
    print("策略波动率", ey.annual_volatility(data_ret.values, period='daily'))
    # 计算累积收益率 
    n = len(data) 
    RcA = (data.收盘[n-1] - data.收盘[0])/data.收盘[0] 
    RcB = (bench.收盘[n-1] - bench.收盘[0])/bench.收盘[0] 
    print("累积收益率", RcA, RcB) 
    # 计算年化收益率
    RaA = pow(1 + RcA, days/n) - 1 
    RaB = pow(1 + RcB, days/n) - 1
    print("年化收益率", RaA, RaB) 
    # 计算最大回撤值 
    MDA = ((data.收盘.cummax() - data.收盘)/data.收盘.cummax()).max() 
    MDB = ((bench.收盘.cummax() - bench.收盘)/bench.收盘.cummax()).max() 
    print("最大回撤:", MDA, MDB) 
    # 计算β值 
    # covAB = np.cov(data_ret, bench_ret)
    covAB = data_ret.cov(bench_ret)
    print("协方差", covAB) 
    varB = np.var(bench_ret, ddof = 1) 
    beta = covAB/varB 
    print("策略β值:", beta)
    # 计算阿尔法值
    # 用empyrical计算α、β值
    beta3 = ey.beta(data_ret.values, bench_ret.values, risk_free = rf)
    alpha = ey.alpha(data_ret.values, bench_ret.values, risk_free = rf, annualization = 1, _beta = beta3)
    print("empyrical计算的α值", alpha*days, "β值", beta3)
    # 计算α值 
    x = data_ret.values
    y = bench_ret.values
    b, a, r_value, p_value, std_err = stats.linregress(x, y) 
    alpha = round(a*days, 4) 
    beta2 = round(b*days, 4) 
    print("α:", alpha, "β:", beta2, "两个β值比值", beta2/beta, "r值", r_value, "p值", p_value) 
    # 另一种计算α的方法 
    alpha = RaA - (rf + beta*(RaB - rf))
    print("另一种方法计算α", alpha) 
    # 计算夏普比率 
    # rf = 0.03 
    rf = (1+rf)**(1/days) - 1.0 
    exReturn = data_ret - rf 
    sharpe = exReturn.mean() / exReturn.std() * np.sqrt(days)
    print("夏普比率:", sharpe) 
    # 计算信息比率 
    ex_return = data_ret - bench_ret
    information = np.sqrt(len(ex_return)) * ex_return.mean() / ex_return.std()
    print("信息比率:", information)
    # empyrical计算信息比率
    info = ey.excess_sharpe(data_ret.values, bench_ret.values)
    print("empyrical计算信息比率", info)
    # 计算索提比例
    sortino = ey.sortino_ratio(returns = data_ret.values)
    print("索提比例", sortino)
    # skew值
    skew = stats.skew(a = data_ret.values)
    print("skew值", skew)
    # calmar值
    calmar = ey.calmar_ratio(data_ret.values)
    print("calmar值", calmar, RaA/MDA)
    
    
if __name__ == "__main__":
    test_risk()
    