# 4. ARIMA模型构建与参数选择

# 导入必要的库
import pandas as pd  # 用于数据处理
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于绘图
import itertools  # 用于生成参数组合
from statsmodels.tsa.arima.model import ARIMA  # ARIMA模型
import warnings  # 用于忽略警告信息
warnings.filterwarnings('ignore')  # 忽略警告信息，使输出更清晰

# 读取AAPL股票数据
data = pd.read_csv('AAPL股票数据.csv', index_col='Date', parse_dates=True)

# 定义评估ARIMA模型的函数
def evaluate_arima_model(X, arima_order):
    """
    评估ARIMA模型在训练集上的性能，并在测试集上进行预测
    
    参数:
        X: 时间序列数据
        arima_order: ARIMA模型的阶数 (p,d,q)
    
    返回:
        包含各种误差指标的字典
    """
    # 准备训练集和测试集（使用80%数据作为训练集）
    train_size = int(len(X) * 0.8)  # 计算训练集大小
    train, test = X[0:train_size], X[train_size:]  # 分割训练集和测试集
    
    # 使用训练集拟合ARIMA模型
    model = ARIMA(train, order=arima_order)  # 创建ARIMA模型
    model_fit = model.fit()  # 拟合模型
    
    # 对测试集进行预测
    predictions = model_fit.forecast(steps=len(test))  # 预测未来len(test)步
    
    # 计算各种误差指标
    mse = np.mean((predictions - test) ** 2)  # 均方误差
    rmse = np.sqrt(mse)  # 均方根误差
    mae = np.mean(np.abs(predictions - test))  # 平均绝对误差
    mape = np.mean(np.abs((predictions - test) / test)) * 100  # 平均绝对百分比误差
    
    # 返回包含误差指标和信息准则的字典
    return {
        'mse': mse, 
        'rmse': rmse, 
        'mae': mae, 
        'mape': mape, 
        'aic': model_fit.aic,  # 赤池信息准则
        'bic': model_fit.bic   # 贝叶斯信息准则
    }

# 网格搜索最优ARIMA参数的函数
def grid_search_arima_params(data, p_values, d_values, q_values):
    """
    通过网格搜索找到最优的ARIMA模型参数
    
    参数:
        data: 时间序列数据
        p_values: 自回归阶数p的候选值列表
        d_values: 差分阶数d的候选值列表
        q_values: 移动平均阶数q的候选值列表
    
    返回:
        results: 所有参数组合的评估结果
        best_cfg: 最优参数组合
    """
    best_score, best_cfg = float('inf'), None  # 初始化最优分数和配置
    results = []  # 用于存储所有结果
    
    # 遍历所有可能的p, d, q组合
    for p, d, q in itertools.product(p_values, d_values, q_values):
        order = (p, d, q)  # 当前ARIMA参数组合
        try:
            # 评估当前参数组合
            result = evaluate_arima_model(data, order)
            results.append((order, result))  # 添加结果到列表
            
            # 使用AIC作为评价标准，更新最优参数
            if result['aic'] < best_score:
                best_score, best_cfg = result['aic'], order
                
            # 打印当前参数组合的评估结果
            print(f'ARIMA{order} - AIC:{result["aic"]:.2f}, BIC:{result["bic"]:.2f}, MSE:{result["mse"]:.4f}')
        except:
            # 如果模型拟合失败，继续下一个参数组合
            continue
            
    return results, best_cfg  # 返回所有结果和最优参数

# 设置参数范围
p_values = range(0, 6)  # 自回归阶数p的候选值: 0,1,2,3,4,5
d_values = [1]  # 差分阶数d的候选值: 1 (根据前面的ADF检验结果)
q_values = range(0, 6)  # 移动平均阶数q的候选值: 0,1,2,3,4,5

# 执行网格搜索，找到最优参数
print("开始ARIMA模型参数网格搜索...")
results, best_cfg = grid_search_arima_params(data['Close'].values, p_values, d_values, q_values)
print(f'最优ARIMA模型参数: {best_cfg}')  # 打印最优参数组合

# 将结果转换为DataFrame，便于分析
results_df = pd.DataFrame([
    {
        'order': f'ARIMA{order}',
        'AIC': result['aic'],
        'BIC': result['bic'],
        'MSE': result['mse'],
        'RMSE': result['rmse'],
        'MAE': result['mae'],
        'MAPE': result['mape']
    }
    for order, result in results
])

# 按AIC值排序
results_df = results_df.sort_values('AIC')
print("\nARIMA模型参数评估结果 (按AIC排序):")
print(results_df.head(10))  # 打印前10个最优参数组合

# 将结果可视化
# 创建热力图，展示不同参数组合的AIC值
try:
    import seaborn as sns  # 用于绘制热力图
    
    # 提取热力图数据
    heatmap_data = []
    for (p, d, q), result in results:
        heatmap_data.append({
            'p': p, 
            'q': q, 
            'AIC': result['aic']
        })
    
    heatmap_df = pd.DataFrame(heatmap_data)
    
    # 将数据透视为矩阵形式
    pivot_table = heatmap_df.pivot_table(index='p', columns='q', values='AIC')
    
    # 绘制热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('ARIMA模型参数选择 - AIC值热力图 (d=1)')
    plt.xlabel('q (移动平均阶数)')
    plt.ylabel('p (自回归阶数)')
    plt.show()
    
    # 绘制不同评估指标的柱状图
    metrics = ['AIC', 'BIC', 'MSE', 'RMSE', 'MAE', 'MAPE']
    top5_models = results_df.head(5)  # 取前5个最优模型
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        # 绘制柱状图
        sns.barplot(x='order', y=metric, data=top5_models, ax=axes[i])
        axes[i].set_title(f'前5个最优模型的{metric}值')
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
        axes[i].grid(True, axis='y')
    
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("seaborn库未安装，跳过热力图绘制")

# 使用最优参数拟合ARIMA模型
best_p, best_d, best_q = best_cfg  # 解包最优参数
best_model = ARIMA(data['Close'], order=best_cfg)  # 创建最优ARIMA模型
best_model_fit = best_model.fit()  # 拟合模型

# 打印模型摘要
print("\n最优ARIMA模型摘要:")
print(best_model_fit.summary())

# 绘制模型拟合结果
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='原始数据')  # 绘制原始数据
plt.plot(best_model_fit.fittedvalues, color='red', label='拟合值')  # 绘制拟合值
plt.title(f'ARIMA{best_cfg}模型拟合结果')
plt.xlabel('日期')
plt.ylabel('股价(美元)')
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测与实际值对比
# 使用前80%的数据训练，后20%的数据测试
train_size = int(len(data) * 0.8)
train_data = data[:train_size]
test_data = data[train_size:]

# 使用训练数据拟合模型
train_model = ARIMA(train_data['Close'], order=best_cfg)
train_model_fit = train_model.fit()

# 预测测试期间的值
forecast = train_model_fit.forecast(steps=len(test_data))

# 绘制预测结果对比图
plt.figure(figsize=(12, 6))
plt.plot(test_data.index, test_data['Close'], label='实际值')  # 绘制实际值
plt.plot(test_data.index, forecast, color='red', label='预测值')  # 绘制预测值
plt.title(f'ARIMA{best_cfg}模型预测结果')
plt.xlabel('日期')
plt.ylabel('股价(美元)')
plt.legend()
plt.grid(True)
plt.show()

# 计算并打印测试集上的预测误差
test_mse = np.mean((forecast - test_data['Close']) ** 2)
test_rmse = np.sqrt(test_mse)
test_mae = np.mean(np.abs(forecast - test_data['Close']))
test_mape = np.mean(np.abs((forecast - test_data['Close']) / test_data['Close'])) * 100

print("\n测试集上的预测误差:")
print(f"MSE: {test_mse:.4f}")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE: {test_mae:.4f}")
print(f"MAPE: {test_mape:.4f}%") 