# 5. 不同窗口预测与评估

# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
from statsmodels.tsa.arima.model import ARIMA  # ARIMA模型
import seaborn as sns  # 用于高级绘图
import warnings  # 用于忽略警告信息
warnings.filterwarnings('ignore')  # 忽略警告，使输出更清晰

# 读取AAPL股票数据
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 假设我们已经确定最优ARIMA模型参数为(1,1,1)
# 在实际应用中，这些参数应来自于前一部分的网格搜索结果
best_p, best_d, best_q = 1, 1, 1  # ARIMA最优参数
best_order = (best_p, best_d, best_q)  # 最优参数组合

# 定义预测窗口
forecast_periods = [3, 7, 30]  # 定义三个预测窗口：3天、7天和30天

# 划分训练集和测试集（使用前80%的数据作为训练集）
train_size = int(len(data) * 0.8)  # 计算训练集大小
train_data = data[:train_size]  # 训练集
test_data = data[train_size:]  # 测试集

# 创建用于存储不同窗口预测结果的字典
results = {}

# 对每个预测窗口进行预测与评估
for period in forecast_periods:
    print(f"\n------- {period}天预测窗口 -------")
    
    # 确保测试数据足够长
    if len(test_data) < period:
        print(f"警告: 测试集长度 ({len(test_data)}) 小于预测期 ({period})")
        continue
    
    # 拟合ARIMA模型
    model = ARIMA(train_data['Close'], order=best_order)  # 创建ARIMA模型
    model_fit = model.fit()  # 拟合模型
    
    # 预测未来period天
    forecast = model_fit.forecast(steps=period)  # 生成预测值
    
    # 获取实际值
    actual = test_data['Close'][:period]  # 获取对应时间窗口的实际值
    
    # 计算误差指标
    mse = np.mean((forecast - actual) ** 2)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    mae = np.mean(np.abs(forecast - actual))  # 平均绝对误差
    mape = np.mean(np.abs((forecast - actual) / actual)) * 100  # 平均绝对百分比误差
    
    # 存储结果
    results[period] = {
        'forecast': forecast,  # 预测值
        'actual': actual,  # 实际值
        'mse': mse,  # 均方误差
        'rmse': rmse,  # 均方根误差
        'mae': mae,  # 平均绝对误差
        'mape': mape  # 平均绝对百分比误差
    }
    
    # 打印评估结果
    print(f"{period}天预测窗口评估结果:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.4f}%")
    
    # 绘制预测结果对比图
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='实际值', color='blue')  # 绘制实际值
    plt.plot(actual.index, forecast, label='预测值', color='red', linestyle='--')  # 绘制预测值
    plt.title(f'ARIMA{best_order} {period}天预测结果', fontsize=15)  # 设置标题
    plt.xlabel('日期')  # 设置x轴标签
    plt.ylabel('股价(美元)')  # 设置y轴标签
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形
    
    # 绘制预测误差
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(actual)), forecast - actual, color='red', alpha=0.7)  # 绘制误差条形图
    plt.axhline(y=0, color='blue', linestyle='-')  # 添加y=0水平线
    plt.title(f'ARIMA{best_order} {period}天预测误差', fontsize=15)  # 设置标题
    plt.xlabel('天数')  # 设置x轴标签
    plt.ylabel('误差(美元)')  # 设置y轴标签
    plt.grid(True, axis='y')  # 显示y轴网格
    plt.show()  # 显示图形

# 创建误差指标的比较图
# 准备比较数据
comparison_data = []
for period, result in results.items():
    comparison_data.append({
        'Window': f'{period}天',  # 预测窗口
        'MSE': result['mse'],  # 均方误差
        'RMSE': result['rmse'],  # 均方根误差
        'MAE': result['mae'],  # 平均绝对误差
        'MAPE': result['mape']  # 平均绝对百分比误差
    })

# 转换为DataFrame
comparison_df = pd.DataFrame(comparison_data)  # 创建比较数据框

# 绘制柱状图比较不同窗口的误差
metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']  # 要比较的指标

fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建2x2子图
axes = axes.flatten()  # 将axes转为一维数组，便于索引

# 为每个指标绘制柱状图
for i, metric in enumerate(metrics):
    sns.barplot(x='Window', y=metric, data=comparison_df, ax=axes[i])  # 绘制柱状图
    axes[i].set_title(f'不同预测窗口的{metric}对比', fontsize=15)  # 设置标题
    axes[i].grid(True, axis='y')  # 显示y轴网格
    if metric == 'MAPE':
        axes[i].set_ylabel(f'{metric} (%)')  # MAPE的y轴标签
    else:
        axes[i].set_ylabel(metric)  # 其他指标的y轴标签

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形

# 绘制不同窗口的预测结果对比
plt.figure(figsize=(15, 8))

# 绘制所有窗口的实际值与预测值
for period in forecast_periods:
    if period in results:
        plt.plot(results[period]['actual'].index, results[period]['actual'], 
                label=f'实际值', color='blue')  # 只绘制一次实际值
        break  # 只需要绘制一次实际值

# 为每个预测窗口绘制预测值
colors = ['red', 'green', 'purple']  # 不同窗口使用不同颜色
for i, period in enumerate(forecast_periods):
    if period in results:
        plt.plot(results[period]['actual'].index, results[period]['forecast'], 
                label=f'{period}天预测', color=colors[i], linestyle='--')

plt.title('不同预测窗口的ARIMA预测结果对比', fontsize=15)  # 设置标题
plt.xlabel('日期')  # 设置x轴标签
plt.ylabel('股价(美元)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制不同窗口的累积误差对比
plt.figure(figsize=(15, 8))

for i, period in enumerate(forecast_periods):
    if period in results:
        # 计算累积绝对误差
        cum_error = np.abs(results[period]['forecast'] - results[period]['actual']).cumsum()
        plt.plot(range(1, period+1), cum_error, label=f'{period}天窗口', color=colors[i], marker='o')

plt.title('不同预测窗口的累积绝对误差对比', fontsize=15)  # 设置标题
plt.xlabel('天数')  # 设置x轴标签
plt.ylabel('累积绝对误差(美元)')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 绘制误差随窗口大小的变化趋势
window_sizes = [period for period in forecast_periods if period in results]  # 有效的窗口大小
error_metrics = {
    'MSE': [results[period]['mse'] for period in window_sizes],  # 各窗口的MSE
    'RMSE': [results[period]['rmse'] for period in window_sizes],  # 各窗口的RMSE
    'MAE': [results[period]['mae'] for period in window_sizes],  # 各窗口的MAE
    'MAPE': [results[period]['mape'] for period in window_sizes]  # 各窗口的MAPE
}

# 为每个误差指标绘制趋势线
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # 创建2x2子图
axes = axes.flatten()  # 将axes转为一维数组，便于索引

for i, (metric, values) in enumerate(error_metrics.items()):
    axes[i].plot(window_sizes, values, marker='o', linestyle='-', linewidth=2)  # 绘制趋势线
    axes[i].set_title(f'{metric}随预测窗口变化趋势', fontsize=15)  # 设置标题
    axes[i].set_xlabel('预测窗口大小(天)')  # 设置x轴标签
    if metric == 'MAPE':
        axes[i].set_ylabel(f'{metric} (%)')  # MAPE的y轴标签
    else:
        axes[i].set_ylabel(metric)  # 其他指标的y轴标签
    axes[i].grid(True)  # 显示网格
    
    # 添加数据标签
    for x, y in zip(window_sizes, values):
        axes[i].annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                        xytext=(0, 10), ha='center')  # 在数据点上方显示数值

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形

# 打印总结表格
print("\n不同预测窗口的误差指标汇总:")
print(comparison_df.to_string(index=False))  # 打印比较数据框

# 分析结果
print("\n预测性能分析结论:")
best_window = comparison_df.loc[comparison_df['MAPE'].idxmin(), 'Window']  # 找出MAPE最小的窗口
print(f"1. MAPE指标最低的是{best_window}预测窗口，说明该窗口的相对预测误差最小。")
print("2. 随着预测窗口的增大，预测误差总体呈现增加趋势，表明ARIMA模型更适合短期预测。")
print("3. 对于ARIMA(1,1,1)模型，在测试数据上的预测能力随时间窗口增加而递减，这反映了模型捕捉长期趋势能力的局限性。") 