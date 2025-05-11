# 2. 平稳性检验与差分处理

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np   # 用于数学运算
import matplotlib.pyplot as plt  # 用于数据可视化
from statsmodels.tsa.stattools import adfuller  # 用于ADF平稳性检验

# 读取AAPL股票数据
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 定义ADF检验函数，用于检验时间序列的平稳性
def adf_test(timeseries):
    # 进行ADF检验并获取结果
    result = adfuller(timeseries)
    
    # 打印ADF检验的结果
    print('ADF统计量: %f' % result[0])  # ADF统计量
    print('p值: %f' % result[1])  # p值，小于0.05表示序列平稳
    print('临界值:')  # 不同显著性水平下的临界值
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    
    # 根据p值判断序列是否平稳
    if result[1] <= 0.05:
        print("结论: 拒绝原假设，数据平稳")  # p值≤0.05，拒绝原假设，序列平稳
    else:
        print("结论: 无法拒绝原假设，数据非平稳")  # p值>0.05，接受原假设，序列非平稳

# 对原始收盘价进行ADF检验
print("原始收盘价序列的ADF检验结果:")
adf_test(data['Close'])  # 检验原始收盘价序列是否平稳

# 计算一阶差分序列
diff_data = data['Close'].diff().dropna()  # 对收盘价进行一阶差分，并删除产生的NaN值

# 对一阶差分序列进行ADF检验
print("\n一阶差分后序列的ADF检验结果:")
adf_test(diff_data)  # 检验差分后的序列是否平稳

# 计算二阶差分序列（如果需要的话）
diff2_data = diff_data.diff().dropna()  # 对一阶差分序列再次差分，得到二阶差分

# 对二阶差分序列进行ADF检验
print("\n二阶差分后序列的ADF检验结果:")
adf_test(diff2_data)  # 检验二阶差分后的序列是否平稳

# 绘制原始序列与差分序列对比图
fig, axes = plt.subplots(3, 1, figsize=(12, 12))  # 创建3个子图

# 绘制原始收盘价序列
axes[0].plot(data['Close'], label='原始收盘价')
axes[0].set_title('原始收盘价序列')
axes[0].grid(True)
axes[0].legend()

# 绘制一阶差分序列
axes[1].plot(diff_data, label='一阶差分序列', color='orange')
axes[1].set_title('一阶差分序列')
axes[1].grid(True)
axes[1].legend()

# 绘制二阶差分序列
axes[2].plot(diff2_data, label='二阶差分序列', color='green')
axes[2].set_title('二阶差分序列')
axes[2].grid(True)
axes[2].legend()

plt.tight_layout()  # 调整子图之间的间距
plt.show()  # 显示图形

# 绘制原始序列与一阶差分序列的直方图对比
fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 创建2个子图

# 绘制原始收盘价的直方图
axes[0].hist(data['Close'], bins=30, alpha=0.7, color='blue')
axes[0].set_title('原始收盘价分布')
axes[0].grid(True)

# 绘制一阶差分序列的直方图
axes[1].hist(diff_data, bins=30, alpha=0.7, color='orange')
axes[1].set_title('一阶差分序列分布')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 计算并打印序列的描述性统计量
print("\n原始收盘价序列的描述性统计量:")
print(data['Close'].describe())

print("\n一阶差分序列的描述性统计量:")
print(diff_data.describe())

print("\n二阶差分序列的描述性统计量:")
print(diff2_data.describe()) 