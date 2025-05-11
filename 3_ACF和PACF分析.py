# 3. ACF和PACF分析确定模型阶数

# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 用于绘制ACF和PACF图
from statsmodels.tsa.stattools import acf, pacf  # 用于计算自相关系数和偏自相关系数

# 读取AAPL股票数据
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 计算一阶差分序列（假设我们已经确定需要一阶差分）
diff_data = data['Close'].diff().dropna()  # 对收盘价序列进行一阶差分并删除NaN值

# 绘制ACF和PACF图，用于确定ARIMA模型的p和q阶数
fig, axes = plt.subplots(2, 1, figsize=(12, 10))  # 创建2x1子图

# 绘制自相关函数(ACF)图
# ACF图用于确定移动平均(MA)模型的阶数q
# 如果ACF在滞后k后截尾（快速衰减到0），则可能需要MA(k)模型
plot_acf(diff_data, ax=axes[0], lags=30)  # 计算并绘制前30阶自相关系数
axes[0].set_title('差分序列的自相关函数(ACF)')  # 设置标题
axes[0].grid(True)  # 添加网格

# 绘制偏自相关函数(PACF)图
# PACF图用于确定自回归(AR)模型的阶数p
# 如果PACF在滞后k后截尾，则可能需要AR(k)模型
plot_pacf(diff_data, ax=axes[1], lags=30)  # 计算并绘制前30阶偏自相关系数
axes[1].set_title('差分序列的偏自相关函数(PACF)')  # 设置标题
axes[1].grid(True)  # 添加网格

plt.tight_layout()  # 调整子图间距
plt.show()  # 显示图形

# 计算并打印ACF和PACF值，以便更精确地分析
acf_values = acf(diff_data, nlags=10)  # 计算前10阶自相关系数
pacf_values = pacf(diff_data, nlags=10)  # 计算前10阶偏自相关系数

print("ACF值 (前10阶):")
for i, value in enumerate(acf_values):
    print(f"Lag {i}: {value:.4f}")  # 打印每个滞后期的自相关系数

print("\nPACF值 (前10阶):")
for i, value in enumerate(pacf_values):
    print(f"Lag {i}: {value:.4f}")  # 打印每个滞后期的偏自相关系数

# 计算自相关系数的置信区间
# 95%置信区间通常为±1.96/sqrt(n)，其中n为样本量
n = len(diff_data)  # 样本数量
conf_interval = 1.96 / np.sqrt(n)  # 计算95%置信区间

print(f"\n95%置信区间: ±{conf_interval:.4f}")  # 打印置信区间

# 分析ACF和PACF图，确定可能的ARIMA模型阶数
print("\n根据ACF和PACF图的分析:")
print("1. 如果ACF图在滞后q期后截尾，而PACF图呈现指数衰减或正弦波形式，则考虑MA(q)模型")
print("2. 如果PACF图在滞后p期后截尾，而ACF图呈现指数衰减或正弦波形式，则考虑AR(p)模型")
print("3. 如果ACF和PACF都呈现指数衰减或正弦波形式，则考虑ARMA(p,q)模型")

# 绘制原始收盘价和差分序列的时间图，帮助理解
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

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

plt.tight_layout()
plt.show()

# 绘制原始ACF和PACF图合并展示
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.stem(range(len(acf_values)), acf_values)  # 使用stem绘制条状图
plt.axhline(y=0, linestyle='--', alpha=0.3, color='black')  # 添加y=0水平线
plt.axhline(y=conf_interval, linestyle='--', alpha=0.3, color='red')  # 添加上置信区间
plt.axhline(y=-conf_interval, linestyle='--', alpha=0.3, color='red')  # 添加下置信区间
plt.title('自相关函数(ACF)')
plt.xlabel('滞后阶数')
plt.ylabel('自相关系数')
plt.grid(True)

plt.subplot(212)
plt.stem(range(len(pacf_values)), pacf_values)  # 使用stem绘制条状图
plt.axhline(y=0, linestyle='--', alpha=0.3, color='black')  # 添加y=0水平线
plt.axhline(y=conf_interval, linestyle='--', alpha=0.3, color='red')  # 添加上置信区间
plt.axhline(y=-conf_interval, linestyle='--', alpha=0.3, color='red')  # 添加下置信区间
plt.title('偏自相关函数(PACF)')
plt.xlabel('滞后阶数')
plt.ylabel('偏自相关系数')
plt.grid(True)

plt.tight_layout()
plt.show() 