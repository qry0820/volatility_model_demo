import pandas as pd
import numpy as np
from arch.univariate import arch_model
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import het_arch
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
import os
import stock_data
import warnings

import time

warnings.filterwarnings('ignore')


# 定义VolatilityModel类
class VolatilityModel:

    def __init__(self, stock_data, model_type: str = 'auto'):

        self.stock_data = stock_data
        # 载入数据
        if self.stock_data.data is None:
            self.stock_data.load_data()
        
        # 计算对数收益率序列
        self.log_return = self.stock_data.calculate_log_return(freq = '1m')

        # 计算对数收益率波动率序列
        self.log_return_volatility = self.stock_data.calculate_log_return_volatility(freq = '1m')

        # 定义模型种类
        self.model_type = model_type.lower()
        if self.model_type not in ['auto', 'arch', 'garch']:    # TODO:  'egarch', 'figarch', 'harch'
            raise ValueError('Invalid model type.')
        
        # 定义最佳阶数
        self.best_order = None

        # 定义模型拟合结果
        self.fit_model_result = None

        # 定义模型验证结果
        self.standardized_residual_is_white_noise = None

        # 定义模型预测结果
        self.predict_result = None

    # 对对数收益率序列作图
    def plot_log_return(self):
            
        # 作图
        log_return_for_plot = self.log_return.copy().dropna()
        plt.plot(log_return_for_plot)
        plt.xlabel('Date')
        plt.ylabel('Log Return')
        plt.grid(True)
        plt.title('Log Return')
        plt.show()

        return
    
    # 作对数收益率序列的ACF图和PACF图
    # 作对数收益率序列平方的ACF图和PACF图
    def plot_acf_pacf(self, lags = 36):
        
        # 作对数收益率序列残差的ACF图和PACF图
        log_return_for_plot = self.log_return.copy().dropna()
        fig = plt.figure(figsize = (12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(log_return_for_plot, lags = lags, ax = ax1, title = 'Autocorrelation of Log Returns')
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(log_return_for_plot, lags = lags, ax = ax2, title = 'Partial Autocorrelation of Log Returns')
        plt.show()

        # 作对数收益率序列平方的ACF图和PACF图
        log_return_squared_for_plot = log_return_for_plot * log_return_for_plot
        fig = plt.figure(figsize = (12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(log_return_squared_for_plot, lags = lags, ax = ax1, title = 'Autocorrelation of Squared Log Returns')
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(log_return_squared_for_plot, lags = lags, ax = ax2, title = 'Partial Autocorrelation of Squared Log Returns')
        plt.show()

        return

    # 平稳性检验
    # ADF检验
    def check_stationary(self, reject_null_hypothesis_threshold = 0.05):
        
        # ADF检验
        log_return_for_test = self.log_return.copy().dropna()
        adf_result = adfuller(log_return_for_test)

        # 提取重要结果
        test_statistic = adf_result[0]
        p_value = adf_result[1]

        # 输出检验结果
        print('ADF检验结果：')
        print('Test Statistic: ', test_statistic)
        print('p-value: ', p_value)
        print('Critical Values: ')
        for key, value in adf_result[4].items():
            print('\t%s: %.3f' % (key, value))
        
        # 判断检验结果是否拒绝原假设
        
        if p_value < reject_null_hypothesis_threshold:
            reject_null_hypothesis = True
            print('原假设在%.2f的显著性水平下被拒绝，序列平稳！' % reject_null_hypothesis_threshold)
            
        else:
            reject_null_hypothesis = False
            print('原假设在%.2f的显著性水平下不能被拒绝，序列不平稳！' % reject_null_hypothesis_threshold)

        # 整理并返回结果
        adf_result_dict = {'test_statistic': test_statistic, 'p_value': p_value}

        return reject_null_hypothesis, adf_result_dict
    
    # 白噪声检验
    def check_white_noise(self, lags = 12, reject_null_hypothesis_threshold = 0.05):
        
        # Ljung-Box白噪声检验
        log_return_for_test = self.log_return.copy().dropna()
        lb_result = sm.stats.diagnostic.acorr_ljungbox(log_return_for_test, lags = lags)

        # 提取重要结果
        print('Ljung-Box白噪声检验结果：')
        print(lb_result, end = '\n\n')

        # 判断检验结果是否拒绝原假设
        reject_null_hypothesis_dict = {}
        for i in lb_result.index:
            print('当lags = ' + str(i) + '时，', end = '')
            if lb_result.loc[i, 'lb_pvalue'] < reject_null_hypothesis_threshold:
                reject_null_hypothesis_dict[i] = True
                print('原假设在%.2f的显著性水平下被拒绝，序列不是白噪声！' % reject_null_hypothesis_threshold)
            
            else:
                reject_null_hypothesis_dict[i] = False
                print('原假设在%.2f的显著性水平下不能被拒绝，序列是白噪声！' % reject_null_hypothesis_threshold)
            
        return reject_null_hypothesis_dict, lb_result

    # 异方差性检验
    def check_arch_effect(self, reject_null_hypothesis_threshold = 0.05):

        # ARCH检验
        log_return_for_test = self.log_return.copy().dropna()
        # 计算残差
        log_return_residual = log_return_for_test - log_return_for_test.mean()
        arch_test_result = het_arch(log_return_residual)
        lagrange_multiplier, p_value, f_statistic, f_p_value = arch_test_result[0], arch_test_result[1], arch_test_result[2], arch_test_result[3]
        print('ARCH检验结果：')
        print('Lagrange Multiplier Statistic: ', lagrange_multiplier)
        print('p-value: ', p_value)
        print('F-statistic: ', f_statistic)
        print('F-statistic p-value: ', f_p_value)

        # 判断检验结果是否拒绝原假设
        if p_value < reject_null_hypothesis_threshold:
            reject_null_hypothesis = True
            print('原假设在%.2f的显著性水平下被拒绝，序列存在ARCH效应！' % reject_null_hypothesis_threshold, end = '\n\n')
        else:
            reject_null_hypothesis = False
            print('原假设在%.2f的显著性水平下不能被拒绝，序列不存在ARCH效应！' % reject_null_hypothesis_threshold, end = '\n\n')
        
        # 整理并返回结果
        arch_test_result_dict = {'lagrange_multiplier': lagrange_multiplier, 'p_value': p_value, 'f_statistic': f_statistic, 'f_p_value': f_p_value}

        return reject_null_hypothesis, arch_test_result_dict
    
    # 模型选择
    def select_model(self):

        if self.model_type != 'auto':
            return self.model_type
        else:
            return

    # 选阶
    def select_order(self, method = 'all_sample', metric = 'aic', window_size = 24, predict_horizon = 1):

        log_return_for_order_selection = self.log_return.copy().dropna() * 100

        if method == 'all_sample':

            order_aic_dict = {}
            
            for each_order in range(1, 25):
                am = arch_model(log_return_for_order_selection, vol = 'Arch', p = each_order)
                res = am.fit(disp = False)
                aic = res.aic
                order_aic_dict[each_order] = aic

            if metric == 'aic':
                # 选出AIC最小的阶
                best_order = min(order_aic_dict, key = order_aic_dict.get)
                print('AIC最小的阶为：' + str(best_order), end = '\n\n')
            
            # TODO: 其他选阶方法
            
            # 保存最优阶
            self.best_order = best_order

            return best_order

        elif method == 'rolling_window':
            
            # 设定TimeSeriesSplit
            tscv = TimeSeriesSplit(max_train_size = window_size, test_size = predict_horizon, n_splits = (len(self.log_return) - window_size) // predict_horizon)

            best_order_dict = {}
            # 滚动窗口确定每个窗口的最优阶
            for train_index, test_index in tqdm(tscv.split(log_return_for_order_selection)):

                each_window_log_return_for_order_selection = log_return_for_order_selection.iloc[train_index]

                order_aic_dict = {}
                
                for each_order in range(1, window_size + 1):
                    am = arch_model(each_window_log_return_for_order_selection, vol = 'Arch', p = each_order)
                    res = am.fit(disp = False)
                    aic = res.aic
                    order_aic_dict[each_order] = aic

                if metric == 'aic':
                    # 选出AIC最小的阶
                    best_order = min(order_aic_dict, key = order_aic_dict.get)

                # TODO: 其他选阶方法
                    
                # 保存该窗口的最优阶
                best_order_dict[train_index[0]] = best_order
            
            # 保存最优阶
            self.best_order = best_order_dict

            return best_order_dict
        
        else:
            raise ValueError('Invalid method.')

    # 模型拟合
    def fit_model(self, data = None, order = None):

        if data is None:
            log_return_for_fit = self.log_return.copy().dropna() * 100

        else:
            log_return_for_fit = data.copy().dropna() * 100

        if order is None:
            order = self.best_order

        if type(order) != int:
            raise ValueError('Invalid order.')

        if self.select_model() == 'arch':

            # 判断是否满足模型拟合条件
            # if self.check_stationary()[0] == False:
            #     raise ValueError('The time series is not stationary.')
            # elif self.check_white_noise()[0] == True:
            #     raise ValueError('The time series is not white noise.')
            # if self.check_arch_effect()[0] == False:
            #     raise ValueError('The time series does not have ARCH effect.')
            
            # 用给定的阶拟合模型
            am = arch_model(log_return_for_fit, vol = 'Arch', p = order)
            res = am.fit(disp = False)

            # 输出模型拟合结果
            # print(res.summary())

            # # 保存拟合结果
            # self.fit_model_result = res

        return res
    
    # 模型验证
    # TODO: 需对滚动窗口做出改进
    def validate_model(self):
        
        # 如果模型拟合结果为空，则先进行模型拟合
        if self.fit_model_result is None:
            self.fit_model()
        
        # 检验标准化残差是否为白噪声
        # 计算标准化残差
        standardized_residual = self.fit_model_result.resid / self.fit_model_result.conditional_volatility
        
        # 对标准化残差作图
        plt.plot(standardized_residual)
        plt.xlabel('Date')
        plt.ylabel('Standardized Residual')
        plt.grid(True)
        plt.title('Standardized Residual')
        plt.show()

        # 对标准化残差作ACF图和PACF图
        fig = plt.figure(figsize = (12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(standardized_residual, lags = 36, ax = ax1, title = 'Autocorrelation of Standardized Residual')
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(standardized_residual, lags = 36, ax = ax2, title = 'Partial Autocorrelation of Standardized Residual')
        plt.show()
        
        # 对标准化残差的平方作ACF图和PACF图
        standardized_residual_squared = standardized_residual * standardized_residual
        fig = plt.figure(figsize = (12, 8))
        ax1 = fig.add_subplot(211)
        fig = sm.graphics.tsa.plot_acf(standardized_residual_squared, lags = 36, ax = ax1, title = 'Autocorrelation of Squared Standardized Residual')
        ax2 = fig.add_subplot(212)
        fig = sm.graphics.tsa.plot_pacf(standardized_residual_squared, lags = 36, ax = ax2, title = 'Partial Autocorrelation of Squared Standardized Residual')
        plt.show()

        # 对标准化残差进行白噪声检验
        standardized_residual_lb_result = sm.stats.diagnostic.acorr_ljungbox(standardized_residual, lags = 12)

        # 提取重要结果
        print('对标准化残差的Ljung-Box白噪声检验结果：')
        print(standardized_residual_lb_result, end = '\n\n')

        # 判断检验结果是否拒绝原假设
        reject_null_hypothesis_dict = {}
        for i in standardized_residual_lb_result.index:
            print('当lags = ' + str(i) + '时，', end = '')
            if standardized_residual_lb_result.loc[i, 'lb_pvalue'] < 0.05:
                reject_null_hypothesis_dict[i] = True
                print('原假设在0.05的显著性水平下被拒绝，序列不是白噪声！')
            
            else:
                reject_null_hypothesis_dict[i] = False
                print('原假设在0.05的显著性水平下不能被拒绝，序列是白噪声！')

        # 如果所有检验结果都不能拒绝原假设，则模型验证通过
        if all(reject_null_hypothesis_dict.values()) == False:
            standardized_residual_is_white_noise = True
            print('所有检验结果都不能拒绝原假设，模型验证通过！\n')
        else:
            standardized_residual_is_white_noise = False
            print('有检验结果拒绝原假设，模型验证不通过！\n')
        
        # 保存模型验证结果
        self.standardized_residual_is_white_noise = standardized_residual_is_white_noise

        return standardized_residual_is_white_noise

    # 使用模型预测波动率
    def predict_volatility(self, fit_model_result = None, horizon = 1):

        if fit_model_result is None:
            fit_model_result = self.fit_model_result


        # # 如果未完成模型验证，则先进行模型验证
        # if self.standardized_residual_is_white_noise is None:
        #     self.validate_model()
        
        # # 如果模型验证不通过，则不进行预测
        # if self.standardized_residual_is_white_noise == False:
        #     raise ValueError('The model is not valid.')
        
        # 用模型预测下一期波动率
        predict_result = np.sqrt(fit_model_result.forecast(horizon = horizon).variance) / 100

        return predict_result

    # 滚动窗口回测
    # 此处假设test_size = 1
    # TODO: 此处需要用到fit_model和predict，以上的函数可能需要更改
    def backtest(self, train_size = 24, test_size = 1, best_order_method = 'all_sample'):
        

        # 设定TimeSeriesSplit
        tscv = TimeSeriesSplit(max_train_size = train_size, test_size = test_size, n_splits = (len(self.log_return) - train_size) // test_size)

        log_return_for_backtest = self.log_return.copy().dropna()
        # 定义缓存预测结果的列表
        predict_result_list = []
        # 滚动窗口回测
        for train_index, test_index in tqdm(tscv.split(log_return_for_backtest)):

            train_data = log_return_for_backtest.iloc[train_index]

            # 训练模型
            if best_order_method == 'all_sample':
                best_order = self.best_order

            elif best_order_method == 'rolling_window':
                best_order = self.best_order[train_index[0]]
            
            res = self.fit_model(data = train_data, order = best_order)

            # 预测波动率
            predict_result = self.predict_volatility(fit_model_result = res, horizon = test_size)

            # 将预测结果保存至列表
            predict_result_list.append(predict_result)

        # 将预测结果转换为DataFrame
        predict_result_df = pd.concat(predict_result_list, axis = 0).sort_index()
        # 注意：此处假设test_size = 1
        predict_result_df.columns = ['predicted_volatility']

        # 注意：此处的预测结果是未来的波动率
        # 注意：此处假设test_size = 1
        # 所以将时间对齐的方法是将预测结果移动一个单位
        predict_result_df = predict_result_df.shift(1)

        # TODO: 如果test_size不等于1，需要修改此处的时间对齐方法

        # 将预测结果与实际波动率合并
        predict_result_df = pd.concat([predict_result_df, self.log_return_volatility], axis = 1).sort_index().dropna()

        # TODO: 计算预测误差

        # 作图
        predict_result_df.plot(grid = True, title = 'Backtest Result')
        # 暂存图像
        plt.savefig('backtest_result_rolling_window.png')
        plt.show()

        return




# 测试用主程序
def main():
    stockdata = stock_data.StockData('AAPL', os.path.join(os.getcwd(), '1d'))
    model = VolatilityModel(stockdata, 'arch')

    # model.plot_log_return()
    # model.plot_acf_pacf()
    # model.check_stationary()
    # print('')
    # model.check_white_noise()
    # print('')
    model.check_arch_effect()
    model.select_order(method = 'rolling_window')
    # model.fit_model()
    model.backtest(best_order_method = 'rolling_window')
    # model.validate_model()
    # model.predict()

    return

if __name__ == '__main__':
    main()