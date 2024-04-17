# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:21:56 2024

@author: 19352
"""

import requests
from datetime import date,datetime
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests, zipfile, io
import os 
import pickle
import copy
import itertools
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt

from pandas.tseries.offsets import DateOffset
os.chdir(r"D:\research assistant of Prof.LU\回测\backtest_collection")
class Exchange:
    
    def __init__(self, trade_symbols, fee=0.0004, initial_balance=10000):
    
        self.initial_balance = initial_balance #初始的资产
        self.fee = fee #交易费用
        self.trade_symbols = trade_symbols #所有可交易的股票代码
        self.recording=pd.DataFrame(columns=[''])
        self.account = {'RMB':{'realised_profit':0, 'unrealised_profit':0, 'total_asset':initial_balance, 'fee':0, 'leverage':0, 'hold':0}}
        '''
        现金账户
        unrealised_profit：浮盈总计
        hold：总持有股票市值
        leverage：总持有股票市值/总资产
        '''
        
        
        for symbol in trade_symbols:
            self.account[symbol] = {'amount':0, 'hold_price':0, 'value':0, 'price':0, 'realised_profit':0,'unrealised_profit':0,'fee':0}
            '''
            amount:持有数量
            hold_price:持有价格
            price:现价
            realised_profit：
            unrealised_profit:浮盈价格
            '''
            
    def Trade(self, symbol, direction, price, amount):
        
        cover_amount = 0 if direction*self.account[symbol]['amount'] >=0 else min(abs(self.account[symbol]['amount']), amount)
        #判断是否需要先平仓以及需要平仓的仓位
        
        open_amount = amount - cover_amount
        #判断是否需要开仓，开仓的话需要开多少
        if not np.isnan(price):
            self.account['RMB']['realised_profit'] -= price*amount*self.fee #扣除手续费
            self.account['RMB']['fee'] += price*amount*self.fee
            self.account[symbol]['fee'] += price*amount*self.fee

        if cover_amount > 0: #先平仓
            if not np.isnan(price):
                self.account['RMB']['realised_profit'] += -direction*(price - self.account[symbol]['hold_price'])*cover_amount  #利润
                self.account[symbol]['realised_profit'] += -direction*(price - self.account[symbol]['hold_price'])*cover_amount
                
                self.account[symbol]['amount'] -= -direction*cover_amount
                self.account[symbol]['hold_price'] = 0 if self.account[symbol]['amount'] == 0 else self.account[symbol]['hold_price']
            else:
                self.account['RMB']['realised_profit'] += -direction*( 0 )*cover_amount  #利润
                self.account[symbol]['realised_profit'] += -direction*( 0 )*cover_amount
                
                self.account[symbol]['amount'] -= -direction*cover_amount
                self.account[symbol]['hold_price'] = 0 if self.account[symbol]['amount'] == 0 else self.account[symbol]['hold_price']
                
        if open_amount > 0:
            if not np.isnan(price):
                total_cost = self.account[symbol]['hold_price']*direction*self.account[symbol]['amount'] + price*open_amount
                total_amount = direction*self.account[symbol]['amount']+open_amount
                
                self.account[symbol]['hold_price'] = total_cost/total_amount
                self.account[symbol]['amount'] += direction*open_amount
                        
        
    def Buy(self, symbol, price, amount):
        self.Trade(symbol, 1, price, amount)
        
    def Sell(self, symbol, price, amount):
        self.Trade(symbol, -1, price, amount)
        
    def Update(self, open_price): #对资产进行更新
        self.account['RMB']['unrealised_profit'] = 0
        self.account['RMB']['hold'] = 0
        for symbol in self.trade_symbols:
            if not np.isnan(open_price[symbol]):
                self.account[symbol]['unrealised_profit'] = (open_price[symbol] - self.account[symbol]['hold_price'])*self.account[symbol]['amount']
                #（现价-买入价）*持有数量
                self.account[symbol]['price'] = open_price[symbol]
                self.account[symbol]['value'] = abs(self.account[symbol]['amount'])*open_price[symbol]
                #持有数量*现价
                self.account['RMB']['hold'] += self.account[symbol]['value']
                
                self.account['RMB']['unrealised_profit'] += self.account[symbol]['unrealised_profit']

#         self.account['RMB']['total_asset'] = round(self.account['RMB']['realised_profit'] + self.account['RMB']['hold'],6)
#         self.account['RMB']['total_profit'] = round(self.account['RMB']['realised_profit']+ self.account['RMB']['unrealised_profit'],6)
        self.account['RMB']['total_asset'] =round(self.account['RMB']['realised_profit'] + self.initial_balance+self.account['RMB']['unrealised_profit'],6)
        self.account['RMB']['leverage'] = round(self.account['RMB']['hold']/self.account['RMB']['total_asset'],3)
        
    def preUpdate(self,open_price): #在每期调仓前获得当期的可使用资金
        hold_asset=0
        for symbol in self.trade_symbols:
            if not np.isnan(open_price[symbol]):
                unrealised_profit = (open_price[symbol] - self.account[symbol]['hold_price'])*self.account[symbol]['amount']
                #持有数量*现价=价值敞口
                hold_asset += unrealised_profit
        usable_asset= round(self.account['RMB']['realised_profit'] + self.initial_balance+ hold_asset,6)  
        return usable_asset

#测试因子的函数
# BIG change to the func Test -sjk 
# can test two factors now -sjk
def Test(factor, symbols, df_open,factor2=None, freq=1, buy_sign1 = [0,0.1], buy_sign2 = [0,0.1], sell_sign = 0.1, initial_balance=1000000, group=0):
    # 更新：buy_sign从float变为[float（x),float(y)], 做多因子值前(x,y)-sjk
    '''
    factor_df: 因子值的dataframe，index需为pd.datetime类型的日期
    factor2: the second factor_df
    symbols_stock   具有因子值的所有股票的ID，类型为list
    open_df   每个月开盘价的dataframe，index需为pd.datetime类型的日期
    risk_free_file 无风险利率的文件名
    freq  多久重新构造投资组合一次，目前支持的输入有：‘July’-> 每年7月初更新；数字1-12->每1-12个月更新一次
    buy_sign： 做多因子值前百分之几的股票，默认为做多前10%，即0.1。# to update-sjk
    group: 用于区分是否分组，若group=0，则只持有一组资产，若group=1，则说明持有多组资产  -冯婷玉
    返回值：
    1.因子的持仓的资产净值
    '''
    e = Exchange(symbols, fee=0.001, initial_balance=initial_balance)
    res_list = []
    index_list = []
    factor = factor.dropna(how='all')
    buy_symbols = []
    usable_assetlist=[]
    trigger=0



    if freq == 'July': ####对于调仓时间固定为每年7月的因子进行回测
        for idx, row in df_open.iterrows():
            if idx.month!=7:
                continue
            prices = df_open.loc[idx,]
            index_list.append(idx)
            
            if trigger>=1:
                usable_asset=e.preUpdate(prices)
                usable_assetlist.append(usable_asset)
                
            if idx in factor.index:
                trigger+=1
                rrow = factor.loc[idx, ]
                if group== 0 :  #####冯婷玉添加
                    buy_symbols1 =  rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))
                                                 &(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index
                else:  ####冯婷玉添加
                    '''
                    question: specific function of group
                
                    '''
                    #buy_symbols =  rrow.dropna()[(rrow.dropna() > rrow.dropna().quantile(buy_sign-0.1))&(rrow.dropna() < rrow.dropna().quantile(buy_sign))].index
                    #buy_symbols1 = rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))&(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index

                # more changes -sjk
                if not factor2:
                    buy_symbols = buy_symbols1
                else:
                    factor2 = factor2.dropna(how = "all")
                    rrow2 = factor2.loc[idx, ]
                    buy_symbols2 = rrow2.dropna()[(rrow2.dropna() < rrow2.dropna().quantile((1-buy_sign2[0])))
                                                &(rrow2.dropna() > rrow2.dropna().quantile((1-buy_sign2[1])))].index
                    buy_symbols = list(set(buy_symbols1) & set(buy_symbols2))
                ### 

                if len(buy_symbols)!=0:
                    number_buy_symbols=len(buy_symbols)
                    if trigger==1:
                        cash_position=(initial_balance/number_buy_symbols)
                    else:
                        cash_position=(usable_asset/number_buy_symbols)
                        
                    if cash_position<=0:
                        cash_position=0
                        buy_symbols=[]
            # Todo: Perhaps we can use df here but not list 
            for symbol in symbols:
                if symbol in buy_symbols and e.account[symbol]['amount'] <= 0:
                    e.Buy(symbol,prices[symbol],(cash_position/prices[symbol])-e.account[symbol]['amount'])
                if symbol not in buy_symbols and e.account[symbol]['amount'] >= 0:
                    e.Sell(symbol,prices[symbol], e.account[symbol]['amount'])
            e.Update(prices)
            res_list.append([e.account['RMB']['total_asset']])
        output=pd.DataFrame(data=res_list, columns=['total_asset'],index = index_list)
        return output

    else: ####对于调仓时间间隔为1，3，6，9，12个月的因子进行回测
        for idx, row in df_open.iterrows():
            prices = df_open.loc[idx,]
            # index_list.append(idx)
            if trigger>=1:
                months_apart = (idx.year - first_transaction_time.year) * 12 + (idx.month - first_transaction_time.month)
                if months_apart%freq != 0:
                    continue
                usable_asset=e.preUpdate(prices)
                usable_assetlist.append(usable_asset)
            index_list.append(idx)    
            if idx in factor.index:
                rrow = factor.loc[idx, ]
                if group== 0 :  #####冯婷玉添加
                    buy_symbols1 =  rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))
                                                 &(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index
                else:  ####冯婷玉添加
                    '''
                    question: specific function of group
                
                    '''
                    #buy_symbols =  rrow.dropna()[(rrow.dropna() > rrow.dropna().quantile(buy_sign-0.1))&(rrow.dropna() < rrow.dropna().quantile(buy_sign))].index
                    #buy_symbols1 = rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))&(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index
                # more changes -sjk
                if  not factor2:
                    buy_symbols = buy_symbols1
                else:
                    factor2 = factor2.dropna(how = "all")
                    rrow2 = factor2.loc[idx, ]
                    buy_symbols2 = rrow2.dropna()[(rrow2.dropna() < rrow2.dropna().quantile((1-buy_sign2[0])))
                                                &(rrow2.dropna() > rrow2.dropna().quantile((1-buy_sign2[1])))].index
                    buy_symbols = list(set(buy_symbols1) & set(buy_symbols2))
                ### 
                
                if len(buy_symbols)!=0:
                    trigger+=1
                    number_buy_symbols=len(buy_symbols)
                    if trigger==1:
                        cash_position=(initial_balance/number_buy_symbols)
                        first_transaction_time=idx
                    else:
                        cash_position=(usable_asset/number_buy_symbols)
                        
                    if cash_position<=0:
                        cash_position=0
                        buy_symbols=[]

            
            for symbol in symbols:
                if symbol in buy_symbols and e.account[symbol]['amount'] <= 0:
                    e.Buy(symbol,prices[symbol],(cash_position/prices[symbol])-e.account[symbol]['amount'])
                if symbol not in buy_symbols and e.account[symbol]['amount'] >= 0:
                    e.Sell(symbol,prices[symbol], e.account[symbol]['amount'])
            e.Update(prices)
            res_list.append([e.account['RMB']['total_asset']])
            #res_list.append([e.account['RMB']['total_asset'],e.account['RMB']['total_profit']])
        
        #output=pd.DataFrame(data=res_list, columns=['total_asset','total_profit'],index = index_list)
        output=pd.DataFrame(data=res_list, columns=['total_asset'],index = index_list)

    return output

# the new func -sjk
def Doubletest(factor,factor2, symbols, df_open,freq=1, sell_sign = 0.1, initial_balance=1000000,sort1 = 3, sort2 = 3):
    """
    sort1: 第一个因子分几类
    sort2：第二个因子分几类

    return: a df, size: (n, (sort1 * sort2))
    """
    buy1_list = [[i/sort1, (i+1)/sort2] for i in range(sort1)]
    buy2_list = [[i/sort1, (i+1)/sort2] for i in range(sort2)]
    buy_list = list(itertools.product(buy1_list, buy2_list))
    list_t = []
    ls_index = [i for i in range(len(buy_list))]
    for i in ls_index:
        temp = Test(factor=factor, factor2 = factor2, symbols=symbols,df_open = df_open,freq=freq,buy_sign1=buy_list[i][0],buy_sign2 = buy_list[i][1],initial_balance=initial_balance)
        temp.columns = [str(i)]
        list_t.append(temp)
    res_dt = pd.concat(list_t,axis =1)
    return res_dt

def mainbody(factor_df,symbols_stock,open_df,risk_free_file,freq=1,buy_sign=0.1,group=0):
### 添加group变量输入，用于区分是否分组，若group=0，则只持有一组资产，若group=1，则说明持有多组资产 -冯婷玉:  
    '''
    factor_df: 因子值的dataframe，index需为pd.datetime类型的日期
    symbols_stock   具有因子值的所有股票的ID，类型为list
    open_df   每个月开盘价的dataframe，index需为pd.datetime类型的日期
    risk_free_file 无风险利率的文件名
    freq  多久重新构造投资组合一次，目前支持的输入有：‘July’-> 每年7月初更新；数字1-12->每1-12个月更新一次
    buy_sign： 做多因子值前百分之几的股票，默认为做多前10%，即0.1。
    group: 用于区分是否分组，若group=0，则只持有一组资产，若group=1，则说明持有多组资产  -冯婷玉
    返回值：
    1.净资产df
    2.年华收益率
    3.夏普率
    4.最大回测
    '''

    factor_res = Test(factor_df, symbols_stock,open_df,freq=freq,buy_sign=buy_sign,group=group) ####冯婷玉添加

    # factor_res['total_asset'].plot(figsize=(15,6),grid=True).get_figure().savefig('WMYC_all.png')
    i = np.argmax((factor_res['total_asset'].cummax() - factor_res['total_asset'])) # 最大回撤结束的位置 最低的那个位置
    if i == 0:
        j = 0
    else:
        j = np.argmax(factor_res['total_asset'][:i])  # 回撤开始的位置 最高的那个点
    maxdrawdown = (factor_res['total_asset'].cummax() - factor_res['total_asset']).max() # 最大回撤
    maxdrawdown_rate = ((factor_res['total_asset'].cummax() - factor_res['total_asset']) / factor_res['total_asset'].cummax()).max() # 最大回撤率
    drawdown_days = i - j # 回撤持续天数
    # print('最大回撤率：', maxdrawdown_rate)
    # print('回撤持续天数：', drawdown_days)
    ####################
    # 夏普率
    risk_free_file=risk_free_file #无风险收益文件
    risk_free = pd.read_excel(risk_free_file).iloc[2:,:]
    risk_free['Trdmnt'] = pd.to_datetime(risk_free['Trdmnt'])
    risk_free.set_index('Trdmnt',inplace=True)
    df_sharp = pd.merge(factor_res, risk_free, left_index=True, right_index=True,how='left')
    df_sharp['return'] = [ (i/1000000)-1 for i in df_sharp.iloc[:,0]]

    # df_sharp=df_sharp[df_sharp['total_asset']>0].iloc[1:,:]
    # print(type(df_sharp.iloc[-1,2]),type(df_sharp['return']),type(len(df_sharp['return'])))
    if freq == 'July' or freq==12 :
        annual_return=  ((1+df_sharp.iloc[-1,2])**(1/len(df_sharp['return'])))-1
        sharp_ration = (annual_return-df_sharp['Interest'].mean()) / df_sharp['return'].std()

    else:
        annual_return=  (df_sharp.iloc[-1,2]/len(df_sharp['return']))*(12/freq)
        sharp_ration = (annual_return-df_sharp['Interest'].mean()) / (df_sharp['return'].std()*np.sqrt(12/freq))
    # print('夏普率：', sharp_ration)
    
    # sharp_ration = (df_sharp['return'].mean() ) / df_sharp['return'].std()
    # df_sharp=pd.DataFrame()
    # df_sharp['return']=factor_res['total_asset'].pct_change()
    # sharp_ration=df_sharp['return'].mean()  / df_sharp['return'].std()
    # print('夏普率：', sharp_ration)
    
    return factor_res,annual_return,sharp_ration,maxdrawdown_rate

#有效性检验(t/IC)-冯婷玉
def t_test(returns, factor, weight, period, start_date, end_date):
    # 生成空的 dict，存储 t 检验、IC 检验结果
    WLS_params = {}
    WLS_t_test = {}
    IC = {}
 
    for i in range(len(factor.index)-1):
        idx = factor.index[i]
        next_idx = factor.index[i+1]
        Y = returns.loc[next_idx, :]
        X = factor.loc[idx, :]
        w = np.sqrt(weight.loc[idx, :])
        l_X=len(X[X.isnull()==False])
        l_Y=len(Y[Y.isnull()==False])
        if l_X<=10 or l_Y<=10: ###有效值太少则跳过该日期
            continue
        # 处理 X,Y,w 中的缺失值，用均值来替换
        X_mean = X.mean()
        X.fillna(X_mean, inplace=True)
        w_mean = w.mean()
        w.fillna(w_mean, inplace=True)
        Y_mean = Y.mean()
        Y.fillna(Y_mean, inplace=True)

        
        index=X.index
        Y=Y[index]
        w=w[index]
        #w=np.sqrt(weight.loc[idx, :][index])
        #print(w)
        #print('X:',X.isnull().any())
        #print('Y:',Y.isnull().any())
        #print('w:',w.isnull().any())
        # WLS 回归
        wls = sm.WLS(Y, X, weights=w)
        output = wls.fit()
        WLS_params[idx] = output.params[-1]
        WLS_t_test[idx] = output.tvalues[-1]
        
        # IC 检验
        IC[idx] = st.pearsonr(Y, X)[0]
    
    return WLS_params, WLS_t_test, IC

#t检验，IC检验-冯婷玉
def IC_test(df_return,factor_df,df_weight,period,start_date,end_date):
    WLS_params,WLS_t_test,IC = t_test(df_return,factor_df,df_weight,period,start_date,end_date)
    WLS_params = pd.Series(WLS_params)
    WLS_t_test = pd.Series(WLS_t_test)
    IC = pd.Series(IC)
    #t检验结果-冯婷玉
    n = [x for x in WLS_t_test.values if np.abs(x)>1.96]
    t_average=np.sum(np.abs(WLS_t_test.values))/len(WLS_t_test)
    proportion_t=len(n)/float(len(WLS_t_test))
    WLS_t_test.plot(kind='bar',figsize=(20,20))
    # 储存结果-冯婷玉
    plt.savefig(rf'result_STY/{file_name}/{file_name}_t.png')
    #IC检验结果-冯婷玉
    n_1 = [x for x in IC.values if x > 0]
    n_2 = [x for x in IC.values if np.abs(x) > 0.02]
    IC.plot(kind='bar',figsize=(20,8))
    '''
    print('t值序列绝对值平均值——判断因子的显著性是否稳定',np.sum(np.abs(WLS_t_test.values))/len(WLS_t_test))
    print('t值序列绝对值大于1.96的占比——判断因子的显著性是否稳定',len(n)/float(len(WLS_t_test)))
    print ('IC 值序列的均值大小',IC.mean())
    print ('IC 值序列的标准差',IC.std())
    print ('IR 比率（IC值序列均值与标准差的比值）',IC.mean()/IC.std())
    print ('IC 值序列大于零的占比',len(n_1)/float(len(IC)))
    print ('IC 值序列绝对值大于0.02的占比',len(n_2)/float(len(IC)))
    '''
    
    # 储存结果-冯婷玉
    plt.savefig(rf'result_STY/{file_name}/{file_name}_IC.png')
    ###output  csv(后续需要根据多因子需求进行改进)
    IC_result=pd.DataFrame(columns=['factor','t_average','proportion_t>1.96','IC_average', 'IC_sd','IR', 'proportion_IC>0','proportion_IC>0.02'])
    IC_result.loc[len(IC_result.index),:]=[file_name,t_average,proportion_t,IC.mean(),IC.std(),IC.mean()/IC.std(),len(n_1)/float(len(IC)),len(n_2)/float(len(IC))]
    IC_result.to_csv(r'result_STY/IC_result.csv',index=False)
    return 

def remove_outliers(data, lower_percentile=0.01, upper_percentile=0.99):
    quantile_df = data.quantile([lower_percentile, upper_percentile])
    filtered_data = data[(data >= quantile_df.loc[lower_percentile]) & (data <= quantile_df.loc[upper_percentile])]
    return filtered_data

def static(factor_df, open_df, risk_free_file, freq):
    group_data = {}

    for group_num in range(1, 11):
        sign = group_num / 10
        print(sign)
        result = mainbody(factor_df, list(factor_df.columns), open_df, risk_free_file, freq=1, buy_sign=sign)
        net_asset_df = result[0]
        
        # 计算月度回报率-钟媛媛
        monthly_returns = net_asset_df['total_asset'].pct_change().dropna()
        # 计算波动率（年化）
        volatility = monthly_returns.std() * np.sqrt(12) 
    
        group_data[f"Group{group_num}_factor_res"] = net_asset_df
        group_data[f"Group{group_num}_annual_returns"] = result[1]
        group_data[f"Group{group_num}_sharpe_ratio"] = result[2]
        group_data[f"Group{group_num}_maxdrawdown_rate"] = result[3]
        group_data[f"Group{group_num}_volatility"] = volatility   # 保存波动率结果-钟媛媛
    # 加载基准数据
    benchmark = pd.read_excel("index_return.xlsx", parse_dates=['date'])
    # 确保日期格式正确并设置为索引
    benchmark.set_index('date', inplace=True)
    
    # 重采样日数据为月数据，取每月最后一个有效值作为该月的数据
    monthly_benchmark = benchmark['return'].resample('MS').mean()
    
    # 从group_data字典中获取第一组和第十组的净资产数据
    group1_assets = group_data["Group1_factor_res"]["total_asset"]
    group9_assets = group_data["Group9_factor_res"]["total_asset"]
    
    # 将净资产数据对齐到月度数据
    group1_assets.index = pd.to_datetime(group1_assets.index)
    group9_assets.index = pd.to_datetime(group9_assets.index)
    
    # 计算月度回报率
    group1_returns = group1_assets.pct_change().dropna()
    group9_returns = group9_assets.pct_change().dropna()
    
    # 确保索引对齐，如果回报率时间序列不完全重叠，需要进一步的处理
    common_index = group1_returns.index.intersection(group9_returns.index).intersection(monthly_benchmark.index)
    group1_returns = group1_returns.loc[common_index]
    group9_returns = group9_returns.loc[common_index]
    monthly_benchmark_returns = monthly_benchmark.loc[common_index]
    
    # 计算第一组和第十组的回报率比
    group1_div_group9 = (1+group1_returns).cumprod() / (1+group9_returns).cumprod()
    # 计算第一组和基准的回报率比
    group1_div_benchmark = (1+group1_returns).cumprod() / (1+monthly_benchmark_returns).cumprod()
    # 计算第十组和基准的回报率比
    group9_div_benchmark = (1+group9_returns).cumprod() / (1+monthly_benchmark_returns).cumprod()
    
    
    
    # 去除极端值
    group1_div_group9 = remove_outliers(group1_div_group9)
    group1_div_benchmark = remove_outliers(group1_div_benchmark)
    group9_div_benchmark = remove_outliers(group9_div_benchmark)
    
    # 绘制比率曲线
    plt.figure(figsize=(14, 7))
    plt.plot(group1_div_group9.index, group1_div_group9, label='Group 1 / Group 9')
    #plt.plot(group1_div_benchmark.index, group1_div_benchmark, label='Group 1 / Benchmark')
    #plt.plot(group9_div_benchmark.index, group9_div_benchmark, label='Group 9 / Benchmark')
    plt.title('Return Ratios Over Time')
    plt.xlabel('Date')
    plt.ylabel('Ratio')
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig('result_ZYY/G1_div_G10_NetAsset.png')

    result_data = pd.DataFrame(columns=["Annual Returns", "Sharpe Ratio", "Max Drawdown Rate", "Volatility"])
    
    # 遍历所有组，从group_data字典中收集每个指标的数据
    for group_num in range(1, 11):
        # 提取每个指标数据
        annual_returns = group_data[f"Group{group_num}_annual_returns"]
        sharpe_ratio = group_data[f"Group{group_num}_sharpe_ratio"]
        maxdrawdown_rate = group_data[f"Group{group_num}_maxdrawdown_rate"]
        volatility = group_data[f"Group{group_num}_volatility"]
    
        # 将数据添加到DataFrame中
        result_data.loc[f"Group {group_num}"] = [annual_returns, sharpe_ratio, maxdrawdown_rate, volatility]
    
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.tight_layout(pad=6.0)
    
    # 年化收益率
    axes[0, 0].bar(result_data.index, result_data['Annual Returns'], zorder=3)
    axes[0, 0].set_title('Annual Returns')
    axes[0, 0].set_xlabel('Group')
    axes[0, 0].set_ylabel('Annual Returns')
    axes[0, 0].tick_params(axis='x', rotation=45)  
    axes[0, 0].grid(True, zorder=2)  
    
    # 夏普比率
    axes[0, 1].bar(result_data.index, result_data['Sharpe Ratio'], zorder=3)
    axes[0, 1].set_title('Sharpe Ratio')
    axes[0, 1].set_xlabel('Group')
    axes[0, 1].set_ylabel('Sharpe Ratio')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, zorder=2) 
    
    # 最大回撤率
    axes[1, 0].bar(result_data.index, result_data['Max Drawdown Rate'], zorder=3)
    axes[1, 0].set_title('Max Drawdown Rate')
    axes[1, 0].set_xlabel('Group')
    axes[1, 0].set_ylabel('Max Drawdown Rate')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, zorder=2) 
    
    # 波动率
    axes[1, 1].bar(result_data.index, result_data['Volatility'], zorder=3)
    axes[1, 1].set_title('Volatility')
    axes[1, 1].set_xlabel('Group')
    axes[1, 1].set_ylabel('Volatility')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, zorder=2)
    
    # 显示图表
    plt.show()
    fig.savefig('result_ZYY/sharp_table.png')
    return 

##Data preparing
print("begin loading______________")
# a new data loader, much faster -sjk
df1 = pd.read_csv("SueAM_200001202312.csv")  ##因子文件
df3 = pd.read_csv("RemM_200001202312.csv")
df2 = pd.read_csv("close_price.csv",encoding='gbk')  ##月度收益率文件

###########################################################
##TO DO：fill up the name of factor in factor file of csv##
###########################################################

facotr_name='SueM' #excel文件中因子的列名称


####################################
####################################

risk_free_file='Rf.xlsx' #无风险收益文件的名字

file_name='Sue' 
'''
输出文件夹和excel的名称
需提前在当前文件夹下创立名字为result的文件夹，然后在result的文件夹内创建与该file_name相同名字的文件夹
'''

##################
#####mainbody#####
##################
# df1 = df1.drop('Unnamed: 3', axis=1)

df1['Trdmnt'] = pd.to_datetime(df1['Trdmnt'],format='%Y-%m') #format可能需要根据实际excel中时间的类型进行修改
# df1['Trdmnt'] = pd.to_datetime(df1['Trdmnt'])
#df1["Trdmnt"] = df1["Trdmnt"].apply(lambda x: x + eval("DateOffset(days=1)"))
df1['Stkcd'] = df1['Stkcd'].astype(str).str.zfill(6)
# df1['Ami']=df1.groupby('Stkcd')['Ami'].shift(1)

df3['Trdmnt'] = pd.to_datetime(df3['Trdmnt'],format='%b-%y') #format可能需要根据实际excel中时间的类型进行修改
# df1['Trdmnt'] = pd.to_datetime(df1['Trdmnt'])
#df1["Trdmnt"] = df1["Trdmnt"].apply(lambda x: x + eval("DateOffset(days=1)"))
df3['Stkcd'] = df3['Stkcd'].astype(str).str.zfill(6)

##################

df2 = df2[2:]
df2['Stkcd'] = df2['Stkcd'].astype(str).str.zfill(6)
df2.rename(columns = {'Mopnprc': 'open', "Mclsprc": 'close'}, inplace = True )
df2['Trdmnt'] = pd.to_datetime(df2['Trdmnt'])

###################
data = pd.merge(df2,df1, on = ['Stkcd','Trdmnt'], how = 'left')
data = pd.merge(data,df3,on = ['Stkcd','Trdmnt'], how = 'left')

####################
symbols = data['Stkcd'].unique()
data.rename(columns= {'Mopnprc': 'open', "Mclsprc": 'close', "Mretwd": 'ret'}, inplace=True)
data['Trdmnt'] = pd.to_datetime(data['Trdmnt'])
data.drop_duplicates()

for column in data.columns:
    if column not in ["Trdmnt","Stkcd"]:
        data[column] = pd.to_numeric(data[column]).astype(np.float64)

####################
start_date = '2000-01-01'
end_date = '2023-10-01' # needs to check manually to make sure data offactors include this end date.-sjk 
period = '1MS'
  
index_range = pd.date_range(start=start_date, end=end_date, freq=period)

# 使用pivot_table代替循环
def pivot_data(df, value_column):
    return df.pivot_table(values=value_column, index='Trdmnt', columns='Stkcd').reindex(index_range)
    
# 为每个指标构建数据帧
df_factor1 = pivot_data(data, 'SueM')
df_factor2 = pivot_data(data,'RemM')
df_close = pivot_data(data, 'close')
df_open = pivot_data(data, 'open')
df_return = pivot_data(data, 'ChangeRatioM')  #####冯婷玉添加
df_weight = pivot_data(data, 'Msmvosd')   ####冯婷玉添加

df_factor = df_factor1.dropna(how = 'all')
df_factor2 = df_factor2.dropna(how = 'all')
df_close = df_close.dropna(how = 'all')
df_open = df_open.dropna(how = 'all')
df_return = df_return.dropna(how = 'all')####冯婷玉添加
df_weight = df_weight.dropna(how = 'all')####冯婷玉添加

df_close = pd.DataFrame(df_close, dtype = np.float64)
df_open = pd.DataFrame(df_open, dtype = np.float64)
factor_df= pd.DataFrame(df_factor, dtype = np.float64)
factor_df2 = pd.DataFrame(df_factor2, dtype = np.float64)
df_return = pd.DataFrame(df_return, dtype = np.float64)  ####冯婷玉添加
df_weight = pd.DataFrame(df_weight, dtype = np.float64)  ####冯婷玉添加

factor_df=copy.deepcopy(df_factor)
factor_df2 = copy.deepcopy(df_factor2)
open_df=copy.deepcopy(df_open)

###IC——test
IC_test(df_return,factor_df,df_weight,period,start_date,end_date)

###two way sorting
print("begin cal_____________________________")
# test the doublesort-sjk
temp1 = Doubletest(factor_df,factor_df2,list(factor_df.columns),open_df,freq=1)
ax = temp1.plot()  # df.plot() 返回一个Matplotlib的Axes对象
print("begin plot__________")
# 保存图像到文件
fig = ax.get_figure()
fig.savefig('result_SJK/my_dataframe_plot.png')  # 保存为PNG文件
print("end________________")

