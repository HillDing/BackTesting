# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 20:17:32 2024

@author: 19352
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 18:30:45 2024

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
os.chdir(r'D:\research assistant of Prof.LU\ML_literature_review\因子实证\data\factor')
import pickle
import copy



from pandas.tseries.offsets import DateOffset
#%matplotlib inline

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
def Test(factor, symbols, df_open,freq=1,buy_sign = 0.1, sell_sign = 0.1, initial_balance=1000000):
    '''
    factor_df: 因子值的dataframe，index需为pd.datetime类型的日期
    symbols_stock   具有因子值的所有股票的ID，类型为list
    open_df   每个月开盘价的dataframe，index需为pd.datetime类型的日期
    risk_free_file 无风险利率的文件名
    freq  多久重新构造投资组合一次，目前支持的输入有：‘July’-> 每年7月初更新；数字1-12->每1-12个月更新一次
    buy_sign： 做多因子值前百分之几的股票，默认为做多前10%，即0.1。
    
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
                buy_symbols =  rrow.dropna()[rrow.dropna() > rrow.dropna().quantile((1-buy_sign))].index

                if len(buy_symbols)!=0:
                    number_buy_symbols=len(buy_symbols)
                    if trigger==1:
                        cash_position=(initial_balance/number_buy_symbols)
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
                buy_symbols =  rrow.dropna()[rrow.dropna() > rrow.dropna().quantile(buy_sign)].index
                # buy_symbols =  rrow.dropna()[rrow.dropna() < rrow.dropna().quantile(1-buy_sign)].index
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
        #            cash_position=(initial_balance/number_buy_symbols)
                    
                #sell_symbols = row.sort_values().dropna()[-N:].index
        
        #         number_buy_symbols=len(buy_symbols)
                
        #         cash_position=(value/number_buy_symbols)
            
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
    #     for j in range(len(output.index)):
    #         output.iloc[j,0]=output.iloc[j,0]-1000000*(j+1)
    return output

def mainbody(factor_df,symbols_stock,open_df,risk_free_file,freq=1,buy_sign=0.1):  
    '''
    factor_df: 因子值的dataframe，index需为pd.datetime类型的日期
    symbols_stock   具有因子值的所有股票的ID，类型为list
    open_df   每个月开盘价的dataframe，index需为pd.datetime类型的日期
    risk_free_file 无风险利率的文件名
    freq  多久重新构造投资组合一次，目前支持的输入有：‘July’-> 每年7月初更新；数字1-12->每1-12个月更新一次
    buy_sign： 做多因子值前百分之几的股票，默认为做多前10%，即0.1。
    
    返回值：
    1.净资产df
    2.年华收益率
    3.夏普率
    4.最大回测
    '''


    
    factor_res = Test(factor_df, symbols_stock,open_df,freq=freq,buy_sign=buy_sign)
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


'''
此处往上的内容不需要修改
'''
##################################################



###dataloading

'''
此处需要按不同因子更新

'''
df1 = pd.read_csv("AmiM_200007202312.csv")  ##因子文件
df2 = pd.read_csv("close price.csv",encoding='gbk')  ##月度收益率文件
facotr_name='Ami' #excel文件中因子的列名称
risk_free_file='Rf.xlsx' #无风险收益文件的名字

file_name='Ami' ##输出文件夹和excel的名称，需提前在当先文件夹下创立名字为result的文件夹，然后在result的文件夹内创建与该file_name相同名字的文件夹


##################
#####mainbody#####
##################
# df1 = df1.drop('Unnamed: 3', axis=1)
# df1['Trdmnt'] = pd.to_datetime(df1['Trdmnt'],format='%b-%y') #format可能需要根据实际excel中时间的类型进行修改
df1['Trdmnt'] = pd.to_datetime(df1['Trdmnt'])
#df1["Trdmnt"] = df1["Trdmnt"].apply(lambda x: x + eval("DateOffset(days=1)"))
df1['Stkcd'] = df1['Stkcd'].astype(str).str.zfill(6)
df1['Ami']=df1.groupby('Stkcd')['Ami'].shift(1)

##################

df2 = df2[2:]
df2['Stkcd'] = df2['Stkcd'].astype(str).str.zfill(6)
df2.rename(columns = {'Mopnprc': 'open', "Mclsprc": 'close'}, inplace = True )
df2['Trdmnt'] = pd.to_datetime(df2['Trdmnt'])

###################
data = pd.merge(df2,df1, on = ['Stkcd','Trdmnt'], how = 'left')


####################
symbols = set(data['Stkcd'])
data.rename(columns= {'Mopnprc': 'open', "Mclsprc": 'close', "Mretwd": 'ret'}, inplace=True)
data['Trdmnt'] = pd.to_datetime(data['Trdmnt'])
data.drop_duplicates()
####################
start_date = '2000-01-01'
end_date = '2023-12-01'
period = '1MS'
df_dict = {}
df_factor = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())
df_close = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())
df_open = pd.DataFrame(index = pd.date_range(start = start_date, end = end_date, freq = period), columns = df_dict.keys())

for symbol in symbols:
    df_s = data[data['Stkcd'] == symbol]
    df_s.set_index(keys='Trdmnt',inplace=True)
    df_s=df_s[~df_s.index.duplicated()]
    if not df_s.empty:
        df_factor[symbol] = df_s[facotr_name]
        df_close[symbol] = df_s['close']
        df_open[symbol] = df_s['open']
        


df_factor = df_factor.dropna(how = 'all')
df_close = df_close.dropna(how = 'all')
df_open = df_open.dropna(how = 'all')
df_close = pd.DataFrame(df_close, dtype = np.float64)
df_open = pd.DataFrame(df_open, dtype = np.float64)
factor_df= pd.DataFrame(df_factor, dtype = np.float64)

factor_df=copy.deepcopy(df_factor)
open_df=copy.deepcopy(df_open)

#####下列为创建不同因子
'''
因子变量创建遵循下列命名规则：
因子前缀=因子名+频率+A(B)+p(可选)+1(可选)
其中：
频率：投资组合更新的频率，目前
A（B）指代构建因子所用财务报表类型：A为合并报表，B为母公司报表
p(可选):因子前缀中具有p则代表做多前20%，不具有P为做多前10%。（记得检查因子名是否有字母p，需要保证p在因子名称中没有p，否则需要修改后续输出excel时的if控制开关）
1(可选)：有1为2010-2023的，无1为2000-2023的

举例：Sue6Ap1
Sue 是因子名
6  代表每隔6个月开一次仓
A  代表构建因子用的是合并报表
p  代表做多前20%
1  代表从2010年开始构建投资组合
'''
# list_factor=['SueA','SueAp','SueA1','SueAp1','Sue3A','Sue3Ap','Sue3A1','Sue3Ap1','Sue6A','Sue6Ap','Sue6A1','Sue6Ap1','Sue9A','Sue9Ap','Sue9A1','Sue9Ap1','Sue12A','Sue12Ap','Sue12A1','Sue12Ap1']
list_factor=['Ami','Amip','Ami1','Amip1']
'''
上面这个列表需要按照因子变量命名规则创建一个因子名称列表
并根据因子变量名称修改下面的各变量前缀

*****如未正确命名，则后续程序无法自动运行
'''
Ami_factor_res,Ami_annual_return,Ami_sharp_ration,Ami_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=1,buy_sign=0.1)
Amip_factor_res,Amip_annual_return,Amip_sharp_ration,Amip_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=1,buy_sign=0.2)
Ami1_factor_res,Ami1_annual_return,Ami1_sharp_ration,Ami1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=1,buy_sign=0.1)
Amip1_factor_res,Amip1_annual_return,Amip1_sharp_ration,Amip1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=1,buy_sign=0.2)

# Sue3A_factor_res,Sue3A_annual_return,Sue3A_sharp_ration,Sue3A_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=3,buy_sign=0.1)
# Sue3Ap_factor_res,Sue3Ap_annual_return,Sue3Ap_sharp_ration,Sue3Ap_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=3,buy_sign=0.2)
# Sue3A1_factor_res,Sue3A1_annual_return,Sue3A1_sharp_ration,Sue3A1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=3,buy_sign=0.1)
# Sue3Ap1_factor_res,Sue3Ap1_annual_return,Sue3Ap1_sharp_ration,Sue3Ap1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=3,buy_sign=0.2)

# Sue6A_factor_res,Sue6A_annual_return,Sue6A_sharp_ration,Sue6A_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=6,buy_sign=0.1)
# Sue6Ap_factor_res,Sue6Ap_annual_return,Sue6Ap_sharp_ration,Sue6Ap_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=6,buy_sign=0.2)
# Sue6A1_factor_res,Sue6A1_annual_return,Sue6A1_sharp_ration,Sue6A1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=6,buy_sign=0.1)
# Sue6Ap1_factor_res,Sue6Ap1_annual_return,Sue6Ap1_sharp_ration,Sue6Ap1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=6,buy_sign=0.2)

# Sue9A_factor_res,Sue9A_annual_return,Sue9A_sharp_ration,Sue9A_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=9,buy_sign=0.1)
# Sue9Ap_factor_res,Sue9Ap_annual_return,Sue9Ap_sharp_ration,Sue9Ap_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=9,buy_sign=0.2)
# Sue9A1_factor_res,Sue9A1_annual_return,Sue9A1_sharp_ration,Sue9A1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=9,buy_sign=0.1)
# Sue9Ap1_factor_res,Sue9Ap1_annual_return,Sue9Ap1_sharp_ration,Sue9Ap1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=9,buy_sign=0.2)

# Sue12A_factor_res,Sue12A_annual_return,Sue12A_sharp_ration,Sue12A_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=12,buy_sign=0.1)
# Sue12Ap_factor_res,Sue12Ap_annual_return,Sue12Ap_sharp_ration,Sue12Ap_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df,risk_free_file,freq=12,buy_sign=0.2)
# Sue12A1_factor_res,Sue12A1_annual_return,Sue12A1_sharp_ration,Sue12A1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=12,buy_sign=0.1)
# Sue12Ap1_factor_res,Sue12Ap1_annual_return,Sue12Ap1_sharp_ration,Sue12Ap1_maxdrawdown_rate=mainbody(factor_df,list(factor_df.columns),open_df[open_df.index>=pd.to_datetime('20091231')],risk_free_file,freq=12,buy_sign=0.2)
####################

'''
下面部分代码分别为输出excel结果和输出图片
'''
result=pd.DataFrame(columns=['因子','做多比例','股票数量','回测时间','年化收益率','年化夏普率','最大回撤'])
time_period=['2000-2023','2010-2023']
# number_transaction_day=261
for i in list_factor:
    if 'p' in i:     
        long_ratio='20%'
    else:
        long_ratio='10%'
        
    if i[-1]=='1': ## 有1为2010-2023的，无1为2000-2023的
        result.loc[len(result.index),:]=[f'{i}',long_ratio,len(factor_df.columns),time_period[1],eval(f'{i}_annual_return'),
                                    eval(f'{i}_sharp_ration'),eval(f'{i}_maxdrawdown_rate')]
        eval(f'{i}_factor_res["total_asset"]').plot(figsize=(15,6),grid=True,title=f'Net Asset Value:{i}').get_figure().savefig(rf'./result/{file_name}/{i}.png')
        plt.close()
    else:
        result.loc[len(result.index),:]=[f'{i}',long_ratio,len(factor_df.columns),time_period[0],eval(f'{i}_annual_return'),
                                    eval(f'{i}_sharp_ration'),eval(f'{i}_maxdrawdown_rate')]
        eval(f'{i}_factor_res["total_asset"]').plot(figsize=(15,6),grid=True,title=f'Net Asset Value:{i}').get_figure().savefig(rf'./result/{file_name}/{i}.png')
        plt.close()
# result.to_csv(f'{facotr_name}_result_.csv',encoding='utf_8_sig',index=False)  
result.to_csv(rf'./result/{file_name}_result_.csv',encoding='utf_8_sig',index=False)  

###输出总图片（所有因子叠加在一起的图片）
# all_return=eval(f'{list_factor[0]}_factor_res')
all_return=eval(f'{i}_factor_res')
a=0
columns_name=[list_factor[0]]
for i in list_factor: # 第一个因子已经merge到all_return中，因此只需从第二个开始merge
    all_return=pd.merge(all_return, eval(f'{i}_factor_res'),right_index=True,left_index=True,how='left')
    columns_name.append(list_factor[a])
    all_return.columns=columns_name
    a+=1

all_return.plot(figsize=(15,6),grid=True,title='Net Asset Value')
plt.legend(loc='best', labels=list_factor)
plt.savefig(rf'./result/{file_name}/{file_name}_All.png')
plt.close()

