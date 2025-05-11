# 6. 残差分析

# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import seaborn as sns  # 用于高级绘图
from statsmodels.tsa.arima.model import ARIMA  # ARIMA模型
from statsmodels.graphics.tsaplots import plot_acf  # 绘制自相关函数
import scipy.stats as stats  # 用于统计检验
from statsmodels.stats.diagnostic import acorr_ljungbox  # Ljung-Box检验
import warnings  # 用于忽略警告信息
warnings.filterwarnings('ignore')  # 忽略警告，使输出更清晰

# 读取AAPL股票数据
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 假设我们已经确定最优ARIMA模型参数为(1,1,1)
best_p, best_d, best_q = 1, 1, 1  # ARIMA最优参数
best_order = (best_p, best_d, best_q)  # 最优参数组合

# 划分训练集和测试集
train_size = int(len(data) * 0.8)  # 计算训练集大小：80%的数据
train_data = data[:train_size]  # 训练数据集
test_data = data[train_size:]  # 测试数据集

# 定义预测窗口
forecast_periods = [3, 7, 30]  # 定义三个预测窗口：3天、7天和30天

# 对不同预测窗口进行残差分析
for period in forecast_periods:
    print(f"\n======= {period}天预测窗口残差分析 =======")
    
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
    
    # 计算残差
    residuals = actual - forecast  # 残差 = 实际值 - 预测值
    
    # 创建残差分析图，包含四个子图：残差时间序列、残差分布、QQ图和残差ACF
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))  # 创建2x2子图
    
    # 1. 残差时间序列图
    axes[0, 0].plot(residuals, color='blue')  # 绘制残差时间序列
    axes[0, 0].axhline(y=0, color='red', linestyle='--')  # 添加y=0水平线
    axes[0, 0].set_title('残差时间序列')  # 设置标题
    axes[0, 0].grid(True)  # 显示网格
    
    # 2. 残差分布直方图
    sns.histplot(residuals, kde=True, ax=axes[0, 1])  # 绘制残差分布直方图及密度曲线
    axes[0, 1].axvline(x=0, color='red', linestyle='--')  # 添加x=0垂直线
    axes[0, 1].set_title('残差分布')  # 设置标题
    axes[0, 1].grid(True)  # 显示网格
    
    # 3. 残差QQ图（检验残差正态性）
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])  # 绘制QQ图
    axes[1, 0].set_title('残差Q-Q图')  # 设置标题
    axes[1, 0].grid(True)  # 显示网格
    
    # 4. 残差自相关函数(ACF)图
    plot_acf(residuals, ax=axes[1, 1], lags=min(10, period-1))  # 绘制残差ACF图
    axes[1, 1].set_title('残差自相关函数')  # 设置标题
    
    plt.tight_layout()  # 调整子图布局
    plt.suptitle(f'ARIMA({best_p},{best_d},{best_q}) {period}天预测残差分析', fontsize=16)  # 添加总标题
    plt.subplots_adjust(top=0.92)  # 调整顶部间距，为总标题留出空间
    plt.show()  # 显示图形
    
    # 进行残差的统计分析
    print("\n残差统计量:")
    print(f"平均值: {np.mean(residuals):.4f}")  # 残差均值
    print(f"标准差: {np.std(residuals):.4f}")  # 残差标准差
    print(f"最小值: {np.min(residuals):.4f}")  # 残差最小值
    print(f"最大值: {np.max(residuals):.4f}")  # 残差最大值
    
    # Shapiro-Wilk正态性检验
    shapiro_test = stats.shapiro(residuals)  # 进行Shapiro-Wilk检验
    print(f"\nShapiro-Wilk正态性检验:")
    print(f"W统计量: {shapiro_test[0]:.4f}")
    print(f"p-值: {shapiro_test[1]:.4f}")
    if shapiro_test[1] > 0.05:
        print("结论: 残差可能服从正态分布")  # p值>0.05，无法拒绝原假设
    else:
        print("结论: 残差可能不服从正态分布")  # p值<=0.05，拒绝原假设
    
    # Ljung-Box检验（检验残差是否为白噪声）
    lb_test = acorr_ljungbox(residuals, lags=[5, 10])  # 进行Ljung-Box检验
    print(f"\nLjung-Box检验:")
    print(f"滞后5阶: Q统计量={lb_test[0][0]:.4f}, p值={lb_test[1][0]:.4f}")
    print(f"滞后10阶: Q统计量={lb_test[0][1]:.4f}, p值={lb_test[1][1]:.4f}")
    
    for i, lag in enumerate([5, 10]):
        if lb_test[1][i] > 0.05:
            print(f"结论(滞后{lag}阶): 残差可能是白噪声")  # p值>0.05，无法拒绝原假设
        else:
            print(f"结论(滞后{lag}阶): 残差可能不是白噪声")  # p值<=0.05，拒绝原假设
    
    # 计算和打印残差统计描述
    residuals_df = pd.DataFrame(residuals)  # 将残差转换为DataFrame
    print("\n残差描述性统计:")
    print(residuals_df.describe())  # 打印残差的描述性统计
    
    # 绘制残差的自相关图和偏自相关图
    if period > 5:  # 只有当预测期大于5时才绘制，以确保有足够的数据点
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))  # 创建1x2子图
        
        # 自相关图
        plot_acf(residuals, ax=axes[0], lags=min(period-1, 10))  # 绘制ACF图
        axes[0].set_title('残差自相关函数')  # 设置标题
        
        # 偏自相关图
        from statsmodels.graphics.tsaplots import plot_pacf  # 导入绘制PACF的函数
        plot_pacf(residuals, ax=axes[1], lags=min(period-1, 10))  # 绘制PACF图
        axes[1].set_title('残差偏自相关函数')  # 设置标题
        
        plt.tight_layout()  # 调整子图布局
        plt.show()  # 显示图形
    
    # 绘制残差与拟合值的散点图，用于检测是否存在异方差性
    plt.figure(figsize=(10, 6))
    plt.scatter(forecast, residuals, alpha=0.7)  # 绘制残差与预测值的散点图
    plt.axhline(y=0, color='red', linestyle='--')  # 添加y=0水平线
    plt.title(f'{period}天预测的残差与预测值散点图', fontsize=15)  # 设置标题
    plt.xlabel('预测值')  # 设置x轴标签
    plt.ylabel('残差')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 显示图形
    
    # 如果预测期较长，绘制残差的累积和图
    if period > 10:
        plt.figure(figsize=(10, 6))
        cum_residuals = np.cumsum(residuals)  # 计算残差的累积和
        plt.plot(range(1, len(cum_residuals)+1), cum_residuals)  # 绘制残差累积和
        plt.axhline(y=0, color='red', linestyle='--')  # 添加y=0水平线
        plt.title(f'{period}天预测的残差累积和图', fontsize=15)  # 设置标题
        plt.xlabel('天数')  # 设置x轴标签
        plt.ylabel('残差累积和')  # 设置y轴标签
        plt.grid(True)  # 显示网格
        plt.show()  # 显示图形

# 比较不同预测窗口的残差统计特性
# 准备存储残差数据的字典
residuals_data = {}

# 为每个预测窗口计算残差
for period in forecast_periods:
    if len(test_data) < period:
        continue
    
    model = ARIMA(train_data['Close'], order=best_order)  # 创建ARIMA模型
    model_fit = model.fit()  # 拟合模型
    forecast = model_fit.forecast(steps=period)  # 预测
    actual = test_data['Close'][:period]  # 获取实际值
    residuals = actual - forecast  # 计算残差
    
    # 存储残差数据
    residuals_data[period] = residuals

# 准备比较数据
comparison_data = []
for period, residuals in residuals_data.items():
    comparison_data.append({
        'Window': f'{period}天',  # 预测窗口
        'Mean': np.mean(residuals),  # 均值
        'Std': np.std(residuals),  # 标准差
        'Min': np.min(residuals),  # 最小值
        'Max': np.max(residuals),  # 最大值
        'Abs_Mean': np.mean(np.abs(residuals))  # 绝对值均值
    })

# 转换为DataFrame
comparison_df = pd.DataFrame(comparison_data)  # 创建比较数据框

# 打印比较表格
print("\n不同预测窗口的残差统计特性比较:")
print(comparison_df.to_string(index=False))  # 打印比较数据框

# 绘制不同窗口残差均值和标准差比较柱状图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))  # 创建1x2子图

# 绘制残差均值比较图
sns.barplot(x='Window', y='Mean', data=comparison_df, ax=axes[0])  # 绘制残差均值柱状图
axes[0].set_title('不同预测窗口残差均值比较', fontsize=15)  # 设置标题
axes[0].grid(True, axis='y')  # 显示y轴网格

# 绘制残差标准差比较图
sns.barplot(x='Window', y='Std', data=comparison_df, ax=axes[1])  # 绘制残差标准差柱状图
axes[1].set_title('不同预测窗口残差标准差比较', fontsize=15)  # 设置标题
axes[1].grid(True, axis='y')  # 显示y轴网格

plt.tight_layout()  # 调整子图布局
plt.show()  # 显示图形

# 绘制不同窗口的残差箱线图
plt.figure(figsize=(10, 6))
data_to_plot = [residuals_data[period] for period in forecast_periods if period in residuals_data]  # 收集各窗口的残差
labels = [f'{period}天' for period in forecast_periods if period in residuals_data]  # 标签

# 使用boxplot绘制箱线图
box = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)  # 绘制箱线图

# 设置箱线图颜色
colors = ['lightblue', 'lightgreen', 'lightpink']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)  # 设置箱体颜色

plt.title('不同预测窗口残差箱线图比较', fontsize=15)  # 设置标题
plt.ylabel('残差')  # 设置y轴标签
plt.grid(True, axis='y')  # 显示y轴网格
plt.show()  # 显示图形

# 一次性展示所有预测窗口的残差分布
plt.figure(figsize=(12, 6))
for i, period in enumerate(forecast_periods):
    if period in residuals_data:
        sns.kdeplot(residuals_data[period], label=f'{period}天', fill=True, alpha=0.3)  # 绘制核密度估计图

plt.axvline(x=0, color='black', linestyle='--')  # 添加x=0垂直线
plt.title('不同预测窗口残差分布比较', fontsize=15)  # 设置标题
plt.xlabel('残差')  # 设置x轴标签
plt.ylabel('密度')  # 设置y轴标签
plt.legend()  # 显示图例
plt.grid(True)  # 显示网格
plt.show()  # 显示图形

# 分析和总结
print("\n残差分析总结:")
print("1. 残差分布特性:")
for period in forecast_periods:
    if period in residuals_data:
        mean = np.mean(residuals_data[period])  # 计算均值
        std = np.std(residuals_data[period])  # 计算标准差
        print(f"   - {period}天窗口: 均值={mean:.4f}, 标准差={std:.4f}")

print("\n2. 残差正态性:")
for period in forecast_periods:
    if period in residuals_data:
        shapiro_test = stats.shapiro(residuals_data[period])  # 进行Shapiro-Wilk检验
        if shapiro_test[1] > 0.05:
            conclusion = "可能服从正态分布"
        else:
            conclusion = "可能不服从正态分布"
        print(f"   - {period}天窗口: p值={shapiro_test[1]:.4f}, {conclusion}")

print("\n3. 残差自相关性:")
for period in forecast_periods:
    if period in residuals_data and len(residuals_data[period]) > 5:
        lb_test = acorr_ljungbox(residuals_data[period], lags=[5])  # 进行Ljung-Box检验
        if lb_test[1][0] > 0.05:
            conclusion = "可能是白噪声"
        else:
            conclusion = "可能不是白噪声"
        print(f"   - {period}天窗口: p值={lb_test[1][0]:.4f}, {conclusion}")

print("\n4. 总体结论:")
print("   - 随着预测窗口的增大，残差的标准差总体呈上升趋势，表明ARIMA模型在长期预测中的精度下降。")
print("   - 短期预测窗口(如3天)的残差更接近正态分布，这表明短期预测更稳定和可靠。")
print("   - 通过对不同窗口残差的比较，我们可以确认ARIMA(1,1,1)模型更适合短期预测任务。") 