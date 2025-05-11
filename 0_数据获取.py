# 0. 数据获取

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
import numpy as np  # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import requests  # 用于发送HTTP请求
import time  # 用于控制API请求速率
import os  # 用于文件和路径操作
from datetime import datetime, timedelta  # 用于日期处理

# 设置图表中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 使用Alpha Vantage API获取股票数据
def get_stock_data_alpha_vantage(symbol, api_key, output_size='full', interval='daily'):
    """
    使用Alpha Vantage API获取股票数据
    
    参数:
        symbol: 股票代码，例如'AAPL'
        api_key: Alpha Vantage API密钥
        output_size: 'compact'(最近100个数据点)或'full'(全部数据)
        interval: 'daily', 'weekly', 'monthly'
        
    返回:
        DataFrame: 包含股票历史数据的DataFrame
    """
    print(f"正在从Alpha Vantage获取{symbol}股票数据...")
    
    # 构建API请求URL
    base_url = 'https://www.alphavantage.co/query'
    
    # 设置请求参数
    params = {
        'function': f'TIME_SERIES_{interval.upper()}',
        'symbol': symbol,
        'outputsize': output_size,
        'apikey': api_key,
        'datatype': 'json'  # 可选'json'或'csv'
    }
    
    # 发送API请求
    response = requests.get(base_url, params=params)
    
    # 检查响应状态
    if response.status_code != 200:
        print(f"错误: API请求失败，状态码{response.status_code}")
        return None
    
    # 解析返回的JSON数据
    data = response.json()
    
    # 检查是否有错误消息
    if 'Error Message' in data:
        print(f"错误: {data['Error Message']}")
        return None
    
    # 提取时间序列数据
    if interval == 'daily':
        time_series_key = 'Time Series (Daily)'
    elif interval == 'weekly':
        time_series_key = 'Weekly Time Series'
    elif interval == 'monthly':
        time_series_key = 'Monthly Time Series'
    else:
        print(f"错误: 不支持的时间间隔 {interval}")
        return None
    
    # 检查数据中是否包含时间序列数据
    if time_series_key not in data:
        print(f"错误: 响应中没有找到时间序列数据，可能是API限制")
        if 'Note' in data:
            print(f"API说明: {data['Note']}")
        return None
    
    # 将数据转换为DataFrame
    time_series = data[time_series_key]
    df = pd.DataFrame(time_series).T  # 转置，使日期成为索引
    
    # 将列名重命名为更易于理解的名称
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # 将数据类型转换为数值型
    for col in df.columns:
        df[col] = pd.to_numeric(df[col])
    
    # 将索引转换为日期时间类型
    df.index = pd.to_datetime(df.index)
    df.index.name = 'Date'
    
    # 按日期升序排序
    df = df.sort_index()
    
    print(f"已成功获取{len(df)}条{symbol}股票数据记录")
    return df

# 2. 使用Financial Modeling Prep API获取股票数据
def get_stock_data_fmp(symbol, api_key, from_date='2010-01-01', to_date=None):
    """
    使用Financial Modeling Prep API获取股票历史价格数据
    
    参数:
        symbol: 股票代码，例如'AAPL'
        api_key: FMP API密钥
        from_date: 开始日期，格式'YYYY-MM-DD'
        to_date: 结束日期，格式'YYYY-MM-DD'，默认为当前日期
        
    返回:
        DataFrame: 包含股票历史数据的DataFrame
    """
    print(f"正在从Financial Modeling Prep获取{symbol}股票数据...")
    
    # 如果没有指定结束日期，使用当前日期
    if to_date is None:
        to_date = datetime.now().strftime('%Y-%m-%d')
    
    # 构建API请求URL
    base_url = 'https://financialmodelingprep.com/api/v3/historical-price-full'
    
    # 设置请求参数
    params = {
        'symbol': symbol,
        'from': from_date,
        'to': to_date,
        'apikey': api_key
    }
    
    # 发送API请求
    response = requests.get(f"{base_url}/{symbol}", params=params)
    
    # 检查响应状态
    if response.status_code != 200:
        print(f"错误: API请求失败，状态码{response.status_code}")
        return None
    
    # 解析返回的JSON数据
    data = response.json()
    
    # 检查是否有错误消息或数据为空
    if 'Error Message' in data or 'historical' not in data:
        print(f"错误: 获取数据失败，可能是API限制或股票代码无效")
        return None
    
    # 提取历史价格数据
    historical_data = data['historical']
    
    # 将数据转换为DataFrame
    df = pd.DataFrame(historical_data)
    
    # 将日期列转换为索引
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df.index.name = 'Date'
    
    # 重命名列以匹配Alpha Vantage的命名
    df = df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    })
    
    # 选择我们需要的列
    cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df = df[cols]
    
    # 按日期升序排序
    df = df.sort_index()
    
    print(f"已成功获取{len(df)}条{symbol}股票数据记录")
    return df

# 3. 使用Yahoo Finance获取股票数据
def get_stock_data_yf(symbol, start_date='2010-01-01', end_date=None, interval='1d'):
    """
    使用Yahoo Finance API获取股票数据
    
    参数:
        symbol: 股票代码，例如'AAPL'
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'，默认为当前日期
        interval: 数据频率，'1d'(日),'1wk'(周),'1mo'(月)
        
    返回:
        DataFrame: 包含股票历史数据的DataFrame
    """
    try:
        # 尝试导入yfinance库
        import yfinance as yf
    except ImportError:
        print("错误: 未安装yfinance库，请使用'pip install yfinance'安装")
        return None
    
    print(f"正在从Yahoo Finance获取{symbol}股票数据...")
    
    # 如果没有指定结束日期，使用当前日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # 使用yfinance获取数据
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        # 检查是否成功获取数据
        if df.empty:
            print(f"错误: 未能获取{symbol}的数据，可能是股票代码无效或日期范围错误")
            return None
        
        # 重命名列以保持一致性
        if 'Stock Splits' in df.columns:
            df = df.drop(columns=['Stock Splits', 'Dividends'])
        
        # 确保只保留我们需要的列
        cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        df = df[cols]
        
        print(f"已成功获取{len(df)}条{symbol}股票数据记录")
        return df
        
    except Exception as e:
        print(f"错误: 获取数据时发生异常 - {str(e)}")
        return None

# 4. 合并多个来源的数据并保存到CSV
def combine_and_save_data(symbol, alpha_vantage_key, fmp_key, output_file=None):
    """
    从多个来源获取股票数据，合并后保存到CSV文件
    
    参数:
        symbol: 股票代码，例如'AAPL'
        alpha_vantage_key: Alpha Vantage API密钥
        fmp_key: Financial Modeling Prep API密钥
        output_file: 输出CSV文件路径，默认为'{symbol}股票数据.csv'
    """
    # 设置默认输出文件名
    if output_file is None:
        output_file = f'{symbol}股票数据.csv'
    
    # 获取Alpha Vantage数据
    df_av = get_stock_data_alpha_vantage(symbol, alpha_vantage_key)
    
    # 获取Financial Modeling Prep数据
    df_fmp = get_stock_data_fmp(symbol, fmp_key)
    
    # 获取Yahoo Finance数据
    df_yf = get_stock_data_yf(symbol)
    
    # 创建一个空的数据框列表，用于存储有效的数据框
    dfs = []
    
    # 检查每个数据框是否有效，如果有效则添加到列表中
    if df_av is not None:
        df_av['Source'] = 'Alpha Vantage'
        dfs.append(df_av)
    
    if df_fmp is not None:
        df_fmp['Source'] = 'FMP'
        dfs.append(df_fmp)
    
    if df_yf is not None:
        df_yf['Source'] = 'Yahoo Finance'
        dfs.append(df_yf)
    
    # 检查是否至少有一个有效的数据框
    if not dfs:
        print("错误: 所有数据源都无法获取有效数据")
        return None
    
    # 如果只有一个数据源有效，直接使用该数据
    if len(dfs) == 1:
        combined_df = dfs[0]
        print(f"注意: 只有一个数据源({combined_df['Source'].iloc[0]})提供了有效数据")
    else:
        # 合并多个数据源
        # 首先将所有数据帧合并
        all_data = pd.concat(dfs)
        
        # 然后按日期和来源分组，取平均值
        combined_df = all_data.groupby(level=0).mean()
        combined_df['Volume'] = combined_df['Volume'].astype(int)  # 交易量应为整数
        print(f"已合并{len(dfs)}个数据源的数据")
    
    # 数据清洗
    # 检查并处理缺失值
    if combined_df.isnull().sum().sum() > 0:
        print(f"注意: 检测到{combined_df.isnull().sum().sum()}个缺失值，使用前向填充方法处理")
        combined_df = combined_df.fillna(method='ffill')
    
    # 检查并处理重复值
    duplicate_dates = combined_df.index.duplicated()
    if duplicate_dates.any():
        print(f"注意: 检测到{duplicate_dates.sum()}个重复日期，保留最后一个值")
        combined_df = combined_df[~duplicate_dates]
    
    # 保存到CSV文件
    combined_df.to_csv(output_file)
    print(f"已将{len(combined_df)}条股票数据记录保存到 {output_file}")
    
    # 绘制获取到的数据图表以验证
    plt.figure(figsize=(12, 6))
    plt.plot(combined_df.index, combined_df['Close'], label=f'{symbol}收盘价')
    plt.title(f'{symbol}股票历史收盘价')
    plt.xlabel('日期')
    plt.ylabel('价格(美元)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{symbol}_历史价格图.png')
    plt.show()
    
    return combined_df

# 5. 分别获取数据并进行比较
def compare_data_sources(symbol, alpha_vantage_key, fmp_key, start_date='2020-01-01', end_date=None):
    """
    获取并比较不同数据源的股票数据
    
    参数:
        symbol: 股票代码，例如'AAPL'
        alpha_vantage_key: Alpha Vantage API密钥
        fmp_key: Financial Modeling Prep API密钥
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'，默认为当前日期
    """
    # 如果没有指定结束日期，使用当前日期
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # 获取Alpha Vantage数据
    df_av = get_stock_data_alpha_vantage(symbol, alpha_vantage_key)
    if df_av is not None:
        # 筛选日期范围
        df_av = df_av[(df_av.index >= start_date) & (df_av.index <= end_date)]
    
    # 获取Financial Modeling Prep数据
    df_fmp = get_stock_data_fmp(symbol, fmp_key, from_date=start_date, to_date=end_date)
    
    # 获取Yahoo Finance数据
    df_yf = get_stock_data_yf(symbol, start_date=start_date, end_date=end_date)
    
    # 检查哪些数据源有效
    sources = []
    if df_av is not None:
        sources.append(('Alpha Vantage', df_av))
    if df_fmp is not None:
        sources.append(('FMP', df_fmp))
    if df_yf is not None:
        sources.append(('Yahoo Finance', df_yf))
    
    # 如果没有有效数据源，则退出
    if not sources:
        print("错误: 没有有效的数据源")
        return
    
    # 数据比较和可视化
    plt.figure(figsize=(15, 10))
    
    # 1. 收盘价比较
    plt.subplot(2, 1, 1)
    for name, df in sources:
        plt.plot(df.index, df['Close'], label=f'{name}')
    
    plt.title(f'{symbol}股票收盘价比较')
    plt.xlabel('日期')
    plt.ylabel('收盘价(美元)')
    plt.legend()
    plt.grid(True)
    
    # 2. 交易量比较
    plt.subplot(2, 1, 2)
    for name, df in sources:
        plt.plot(df.index, df['Volume'], label=f'{name}')
    
    plt.title(f'{symbol}股票交易量比较')
    plt.xlabel('日期')
    plt.ylabel('交易量')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{symbol}_数据源比较.png')
    plt.show()
    
    # 计算数据源之间的差异
    if len(sources) > 1:
        print("\n不同数据源收盘价的差异统计:")
        # 选择一个参考数据源
        reference_name, reference_df = sources[0]
        
        for name, df in sources[1:]:
            # 确保只比较共有的日期
            common_dates = reference_df.index.intersection(df.index)
            if len(common_dates) > 0:
                ref_subset = reference_df.loc[common_dates, 'Close']
                df_subset = df.loc[common_dates, 'Close']
                
                # 计算绝对差异和相对差异的统计量
                abs_diff = (df_subset - ref_subset).abs()
                rel_diff = ((df_subset - ref_subset) / ref_subset * 100).abs()
                
                print(f"\n{reference_name} vs {name}:")
                print(f"共有交易日: {len(common_dates)}天")
                print(f"绝对差异统计 (美元):")
                print(f"  平均差异: {abs_diff.mean():.4f}")
                print(f"  最大差异: {abs_diff.max():.4f}")
                print(f"  最小差异: {abs_diff.min():.4f}")
                print(f"  标准差: {abs_diff.std():.4f}")
                
                print(f"相对差异统计 (%):")
                print(f"  平均差异: {rel_diff.mean():.4f}%")
                print(f"  最大差异: {rel_diff.max():.4f}%")
                print(f"  最小差异: {rel_diff.min():.4f}%")
                print(f"  标准差: {rel_diff.std():.4f}%")
                
                # 找出差异最大的日期
                max_diff_date = abs_diff.idxmax()
                print(f"差异最大的日期: {max_diff_date}")
                print(f"  {reference_name}: {ref_subset.loc[max_diff_date]:.4f}")
                print(f"  {name}: {df_subset.loc[max_diff_date]:.4f}")
                print(f"  绝对差异: {abs_diff.loc[max_diff_date]:.4f}美元")
                print(f"  相对差异: {rel_diff.loc[max_diff_date]:.4f}%")
            else:
                print(f"{reference_name}和{name}没有共同的交易日")

# 主程序
if __name__ == "__main__":
    # 设置API密钥
    alpha_vantage_key = "2G35WDR73AU0LHIQ"  # Alpha Vantage API密钥
    fmp_key = "qhylk6wN8OUWTgmLddldoMRPCo59NmBU"  # Financial Modeling Prep API密钥
    
    # 设置股票代码和日期范围
    symbol = "AAPL"  # 苹果公司股票代码
    start_date = "2021-01-01"
    end_date = "2021-12-31"  # 可以设置为None使用当前日期
    
    # 获取并保存数据
    print("\n============= 获取并合并多个数据源的数据 =============")
    df_combined = combine_and_save_data(symbol, alpha_vantage_key, fmp_key)
    
    # 比较不同数据源的数据
    print("\n============= 比较不同数据源的数据 =============")
    compare_data_sources(symbol, alpha_vantage_key, fmp_key, start_date, end_date)
    
    # 如果需要，您可以添加更多股票代码进行测试
    # other_symbols = ["MSFT", "GOOG", "AMZN"]
    # for sym in other_symbols:
    #     combine_and_save_data(sym, alpha_vantage_key, fmp_key) 