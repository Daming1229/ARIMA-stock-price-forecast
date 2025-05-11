# 1. 数据预处理与可视化

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数学运算
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns  # 提供更美观的数据可视化
from statsmodels.tsa.stattools import adfuller  # 用于ADF平稳性检验

# 读取AAPL股票数据，设置Date列为索引，并将其解析为日期格式
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 数据预处理：处理缺失值，使用前向填充法
data = data.fillna(method='ffill')  # 使用前一个有效值填充后续的缺失值

# 定义异常值检测函数，基于滚动窗口的均值和标准差（3σ原则）
def detect_outliers(data, column, window=20, threshold=3):
    # 计算指定列的滚动窗口均值
    rolling_mean = data[column].rolling(window=window).mean()
    # 计算指定列的滚动窗口标准差
    rolling_std = data[column].rolling(window=window).std()
    
    # 确定上下边界：均值±threshold倍标准差
    upper_bound = rolling_mean + threshold * rolling_std
    lower_bound = rolling_mean - threshold * rolling_std
    
    # 识别超出边界的值作为异常值
    outliers = ((data[column] > upper_bound) | (data[column] < lower_bound))
    return outliers

# 绘制收盘价时间序列图
plt.figure(figsize=(12, 6))  # 设置图形大小
plt.plot(data.index, data['Close'], label='AAPL收盘价')  # 绘制收盘价时间序列
plt.title('苹果公司股票收盘价(2014-2017)', fontsize=15)  # 设置图表标题
plt.xlabel('日期')  # 设置x轴标签
plt.ylabel('价格(美元)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 展示图表

# 获取AAPL股票的基本统计信息
print("AAPL股票数据基本统计信息:")
print(data.describe())

# 检查并显示异常值
close_outliers = detect_outliers(data, 'Close')
print(f"\n检测到的收盘价异常值数量: {close_outliers.sum()}")
if close_outliers.sum() > 0:
    print("异常值日期和对应收盘价:")
    print(data[close_outliers][['Close']])

# 绘制收盘价的箱线图，直观展示可能的异常值
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Close'])
plt.title('AAPL股票收盘价箱线图', fontsize=15)
plt.xlabel('收盘价(美元)')
plt.grid(True, axis='y')
plt.show()

# 计算收益率
data['Returns'] = data['Close'].pct_change() * 100  # 计算每日百分比收益率
data['LogReturns'] = np.log(data['Close'] / data['Close'].shift(1)) * 100  # 计算对数收益率

# 绘制收益率分布
plt.figure(figsize=(12, 6))
sns.histplot(data['Returns'].dropna(), kde=True, bins=50)
plt.title('AAPL股票日收益率分布', fontsize=15)
plt.xlabel('日收益率(%)')
plt.ylabel('频率')
plt.grid(True)
plt.show()

# 绘制收盘价和交易量的关系
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(data.index, data['Close'], label='收盘价')
plt.title('AAPL股票收盘价', fontsize=15)
plt.ylabel('价格(美元)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.bar(data.index, data['Volume'], label='交易量', alpha=0.7, color='green')
plt.title('AAPL股票交易量', fontsize=15)
plt.xlabel('日期')
plt.ylabel('交易量')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show() 