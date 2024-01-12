import pandas as pd
import numpy as np
import os

# 定义StockData类
class StockData:

    def __init__(self, ticker, path, start_date = None, end_date = None):

        self.ticker = ticker
        self.path = path
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    # 读取1d数据
    def load_data(self):

        try:
            self.data = pd.read_csv(os.path.join(self.path, self.ticker + '.csv'), index_col = 0)
            self.data.index = pd.to_datetime(self.data.index)

        except:
            print('Error: Data file not found.')

        return
    
    # 展示数据
    def show_data(self):

        try:
            print(self.data)
        
        except:
            print('Error: Data file not found.')
        
        return
    
    # 返回数据
    def get_data(self):
        
        return self.data
    
    # 展示数据的统计信息
    def show_summary(self):

        try:
            print(self.data.describe())
        
        except:
            print('Error: Data file not found.')
        
        return

    # 计算百分比收益率
    def calculate_pct_return(self, freq):
        
        if freq == '1d':
            pct_return = self.data['PX_LAST'].pct_change()
            
        elif freq == '1m':
            pct_return = self.data['PX_LAST'].groupby(pd.Grouper(freq = 'M')).last().pct_change()
        
        pct_return = pct_return.replace([np.inf, -np.inf], np.nan)
        pct_return.name = 'pct_return'

        return pct_return
    
    # 计算对数收益率
    def calculate_log_return(self, freq):

        if freq == '1d':
            log_return = np.log(self.data['PX_LAST']) - np.log(self.data['PX_LAST'].shift(1))
        
        elif freq == '1m':
            monthly_close_data = self.data['PX_LAST'].groupby(pd.Grouper(freq = 'M')).last()
            log_return = np.log(monthly_close_data) - np.log(monthly_close_data.shift(1))
        
        log_return = log_return.replace([np.inf, -np.inf], np.nan)
        log_return.name = 'log_return'
        
        return log_return
    
    # 计算波动率
    def calculate_log_return_volatility(self, freq = '1m'):
        
        # 计算日对数收益率
        log_return = self.calculate_log_return(freq = '1d').ffill().dropna()

        # 计算波动率
        if freq == '1m':

            # 计算月度波动率
            volatility = log_return.groupby(pd.Grouper(freq = 'M')).std()
        
        else:
            print('Error: Invalid frequency.')
            return
        
        volatility = volatility.replace([np.inf, -np.inf], np.nan)
        volatility.name = 'volatility'

        return volatility
    
    