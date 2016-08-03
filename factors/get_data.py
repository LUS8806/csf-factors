#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据模块
从csf数据接口获取分析用的数据
"""

import pandas as pd
import csf
import os
from dateutil.parser import  parse


def get_trade_calendar():
    """
    从trade_cal.csv文件读取交易日历数据
    """
    file_path = os.path.abspath(__file__)
    dir_name = os.path.split(file_path)[0]
    csv_file = os.path.join(dir_name, 'trade_cal.csv')
    trade_cal = pd.read_csv(csv_file, names=['date_time', 'total_day'],
                            index_col=[0], parse_dates=True)
    return trade_cal


def get_stock_industry(codes):
    """
    股票所属数库一级行业
    :param codes: list, 股票代码列表
    :return: 股票与对应行业
    """
    codes_len = len(codes)
    fields = ['code', 'secu_name', 'level1_name', 'level1_code']

    if codes_len > 100:
        cutter = range(0, codes_len, 100)
        cutter.append(codes_len-1)
        dict_cutter = zip(cutter[0:-1], cutter[1:])
        df = pd.DataFrame()
        for i, j in dict_cutter:
            sub_codes = codes[i: j]
            temp = csf.get_stock_industry(sub_codes, field=fields)
            df = pd.concat([df, temp])
        return df

    else:
        return csf.get_stock_industry(codes, field=fields)


def get_benchmark_return(bench_code, dt_index):
    """
    BenchMark收益率
    :param bench_code: str, benchMark代码，如'000300'
    :param dt_index:
    :return:
    """
    st = dt_index[0]
    et = dt_index[-1]
    field = ['close']
    df = csf.get_index_hist_bar(
        index_code=bench_code, start_date=st, end_date=et, field=field)
    price = df['close'].ix[dt_index, :]
    ret = price.pct_change().shift(-1).dropna()
    return ret


def get_raw_factor(factors, index_code, start_date, end_date, freq='M'):
    """
    原始因子值（未经处理过）
    :param factors: str or list, 因子代码"M009006"或因子代码列表["M009006", "M009007"]
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周期，周"W"、月"M"、季"Q"，每个周期的最后一个交易日
    :param filter: dict, 股票筛选
    :return: pd.DataFrame，因子值
    """
    temp = csf.get_stock_factor(factors=factors, index=index_code,
                                start_date=start_date, end_date=end_date, freq=freq)
    df = pd.pivot_table(temp, values='value', index=['date', 'code'], columns=['cd'])
    return df


def get_term_return(index_code, dt_index):
    """
    股票收益率数据
    :param index_code: str, 指数代码，如"000300"
    :param dt_index:
    :return:
    """

    pass


def get_cap_data(index_code, start_date, end_date, freq='M'):

    """
    总市值数据
    :param index_code: str, 指数代码，如"000300"
    :param start_date: str, 开始日期，如"2008-04-30"
    :param end_date: str，结束日期，如"2015-12-31"
    :param freq: str，换仓周日，周"W"、月"M"、季"Q"
    :return: pd.DataFrame，因子值
    """
    return get_raw_factor('M004023', index_code, start_date, end_date, freq)


def get_index_component(index_code, date):
    """
    指数历史成分股
    :param index_code: str, 指数代码'000300'
    :param date: str, 日期'2015-01-10'
    :return: list, 股票代码
    """
    df = csf.get_index_component(index_code, date)
    return df.code.tolist()


def get_stock_lst_date(codes):
    """
    股票首次上市日期
    :param codes: list, 股票代码列表
    :param date: str， 日期'2015-04-30'
    :return:
    """
    field = ['dt']
    df = pd.Series(index=codes)
    for code in codes:
        df['code'] = csf.get_stock_ipo_info(code, field=field).dt[0]
    return df