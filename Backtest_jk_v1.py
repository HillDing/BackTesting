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

from pandas.tseries.offsets import DateOffset

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
def Test(factor,factor2, symbols, df_open,freq=1,buy_sign1 = [0,0.1], buy_sign2 = [0,0.1], sell_sign = 0.1, initial_balance=1000000):
    # 更新：buy_sign从float变为[float（x),float(y)], 做多因子值前(x,y)-sjk
    '''
    factor_df: 因子值的dataframe，index需为pd.datetime类型的日期
    factor2: the second factor_df
    symbols_stock   具有因子值的所有股票的ID，类型为list
    open_df   每个月开盘价的dataframe，index需为pd.datetime类型的日期
    risk_free_file 无风险利率的文件名
    freq  多久重新构造投资组合一次，目前支持的输入有：‘July’-> 每年7月初更新；数字1-12->每1-12个月更新一次
    buy_sign： 做多因子值前百分之几的股票，默认为做多前10%，即0.1。# to update-sjk
    
    返回值：
    1.因子的持仓的资产净值
    '''
    e = Exchange(symbols, fee=0.001, initial_balance=initial_balance)
    res_list = []
    index_list = []
    factor = factor.dropna(how='all')
    factor2 = factor2.dropna(how = "all")
    buy_symbols = []
    usable_assetlist=[]
    trigger=0

    # add a flag -sjk
    flag2 = factor2.empty
    ###

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
                buy_symbols1 = rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))
                                             &(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index
                # more changes -sjk
                if flag2:
                    buy_symbols = buy_symbols1
                else:
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
                buy_symbols1 = rrow.dropna()[(rrow.dropna() < rrow.dropna().quantile((1-buy_sign1[0])))
                                             &(rrow.dropna() > rrow.dropna().quantile((1-buy_sign1[1])))].index
                # more changes -sjk
                if flag2:
                    buy_symbols = buy_symbols1
                else:
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

print("begin loading______________")
# a new data loader, much faster -sjk
df1 = pd.read_csv("/home/np/py_code/Fangzhou Lu RA/framework/backtesting_uncompletely/SueAM_200001202312.csv")  ##因子文件
df3 = pd.read_csv("/home/np/py_code/Fangzhou Lu RA/framework/backtesting_uncompletely/RemM_200001202312.csv")
df2 = pd.read_csv("/home/np/py_code/Fangzhou Lu RA/framework/backtesting_uncompletely/close price.csv",encoding='gbk')  ##月度收益率文件

###########################################################
##TO DO：fill up the name of factor in factor file of csv##
###########################################################

facotr_name='SueM' #excel文件中因子的列名称
pass

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
"""
df_dict = {}
df_factor = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())
df_close = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())
df_open = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())

# Todo: find a better way to modify this, too fucking slow
for symbol in symbols:
    df_s = data[data['Stkcd'] == symbol]
    df_s.set_index(keys='Trdmnt',inplace=True)
    df_s=df_s[~df_s.index.duplicated()]
    if not df_s.empty:
        df_factor[symbol] = df_s[facotr_name]
        df_close[symbol] = df_s['close']
        df_open[symbol] = df_s['open']
"""        
index_range = pd.date_range(start=start_date, end=end_date, freq=period)

# 使用pivot_table代替循环
def pivot_data(df, value_column):
    return df.pivot_table(values=value_column, index='Trdmnt', columns='Stkcd').reindex(index_range)
    
# 为每个指标构建数据帧
df_factor1 = pivot_data(data, 'SueM')
df_factor2 = pivot_data(data,'RemM')
df_close = pivot_data(data, 'close')
df_open = pivot_data(data, 'open')

df_factor = df_factor1.dropna(how = 'all')
df_factor2 = df_factor2.dropna(how = 'all')
df_close = df_close.dropna(how = 'all')
df_open = df_open.dropna(how = 'all')
df_close = pd.DataFrame(df_close, dtype = np.float64)
df_open = pd.DataFrame(df_open, dtype = np.float64)
factor_df= pd.DataFrame(df_factor, dtype = np.float64)
factor_df2 = pd.DataFrame(df_factor2, dtype = np.float64)
factor_df=copy.deepcopy(df_factor)
factor_df2 = copy.deepcopy(df_factor2)
open_df=copy.deepcopy(df_open)
print("begin cal_____________________________")
# test the doublesort-sjk
temp1 = Doubletest(factor_df,factor_df2,list(factor_df.columns),open_df,freq=1)
ax = temp1.plot()  # df.plot() 返回一个Matplotlib的Axes对象
print("begin plot__________")
# 保存图像到文件
fig = ax.get_figure()
fig.savefig('/home/np/py_code/Fangzhou Lu RA/framework/backtesting_uncompletely/my_dataframe_plot.png')  # 保存为PNG文件
print("end________________")