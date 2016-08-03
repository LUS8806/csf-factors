# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from get_data import *
from config import STOCK_FILTER


def mean_abs_deviation(data):
    """
    计算平均绝对离差 Mean Absolute Deviation

    :param data： pd.DataFrame or pd.Series

    :return: float
        mean absolute deviation
    """
    return ((data - data.mean()).abs()).mean()


def stock_filter(codes, date, filter=STOCK_FILTER):
    """
    股票过滤，去除ST、停牌及因子值为空等股票
    :param codes: list, 股票代码列表
    :param date: str， 日期'2015-04-30'
    :param filter: dict, {'ST': True, 'TP': True
    :return:
    """
    # 过滤ST股票
    if filter['ST']:
        st_codes = csf.get_st_stock_today(date).code.tolist()
        codes = [i for i in codes if i not in st_codes]

    # 过滤停牌股票
    if filter['TP']:
        codes = csf.get_stock_sus_today(codes, date).code.tolist()

    # 上市不满N天的股票
    if filter['SLD']:
        num = filter['SLD']
        field = ['dt']
        for code in codes:
            dt = csf.get_stock_ipo_info(code, field=field).dt[0]
            num_s = (parse(date) - parse(dt)).days
            if num_s < num:
                codes.remove(code)
    return codes


def extreme_process(data, num=3, method='mad'):
    """
    去极值处理，极值的判断可以根据标准差或者平均绝对离差（MAD），如果数值超过num个标准差
    （或MAD）则使其等于num个标准差（或MAD）

    :param data: pd.DataFrame or pd.Series
    :param num: int
        超过num个标准差（或MAD）即认为是极值
    :param method: str
        极值的评判标准
            'mad': mean absolute deviation
            'std': standard deviation

    :return pd.DataFrame
        去极值后的数据
    """
    if isinstance(data, pd.Series):
        data = data.to_frame()
    mu = data.mean()
    ind = mean_abs_deviation(data) if method == 'mad' else data.std()

    try:
        ret = data.clip(lower=mu - num * ind, upper=mu + num * ind, axis=1)
    except Exception, e:
        lst = []
        for col in data.columns:
            if data.loc[:, col].isnull().all():
                lst.append(data.loc[:, col])
            else:
                mu = data.loc[:, col].mean()
                ind = mean_abs_deviation(data.loc[:, col]) if method == 'mad' else data.loc[:, col].std()
                lst.append(data.loc[:, col].clip(lower=mu - num * ind, upper=mu + num * ind))
        ret = pd.concat(lst, axis=1)
    return ret


def data_scale(data, cap=None, method='normal'):
    """
    数据标准化处理

    :param data: pd.Series or pd.DataFrame
    :param method: str
        "normal": (x-x.mean())/x.std()
        "cap": (x - cap_weighted_mean of x)/x.std()
        "industry":

    :return: pd.Series or pd.DataFrame
        标准化处理后的数据
    """
    if method == 'normal':
        return (data-data.mean())/data.std()

    elif method == 'cap':
        if isinstance(cap, pd.DataFrame):
            cap = cap.ix[:, 0]
        return data.apply(lambda x: (x - np.sum(x * cap) / cap.sum()) / x.std())

    else:
        raise ValueError


def cut_group(data_, num_group, col_name=None, ascending=False):
    """
    对于给定的数据（普通的DataFrame或Series），按照指定的列（col_name）
    进行排序（升序或降序）并分为num_group组

    :param data_: pd.Series of pd.DataFrame
    :param col_name: str
        根据该列值的大小进行排序
    :param num_group: int
        分成的组数
    :param ascending: Bool
        True: ascending, False: Descending

    :return pd.DataFrame
        在原先的data后增加一列，每个元素对应的是该行对应的组数
    """
    data = data_.copy()
    if isinstance(data, pd.DataFrame):
        data = data.loc[:, col_name]

    data_len = len(data)
    avg_element = data_len // num_group
    remains = data_len % num_group
    each_group = [avg_element] * num_group
    if remains:
        for idx in range(0, remains):
            each_group[idx] += 1
    each_group = np.array(each_group)
    each_group = each_group.cumsum()
    try:
        idx = data.rank(method='first', na_option='bottom', ascending=ascending)
    except:
        print(idx)
        print(data.index[0])
        print(col_name)
        print('error occurred in cut_group')
    groups = pd.Series(index=idx.index)
    start = 0
    for grp, end in enumerate(each_group):
        mask = (idx > start) & (idx <= end)
        groups[mask] = ''.join(['Q', str(grp + 1)])
        # groups[mask] = grp
        start = end
    groups = groups.tolist()
    data['group'] = groups
    return groups


def get_grouped_data(fac_ret_cap, col_name, num_group, ascending):
    """
    对MultiIndex的数据(level=0)进行分组

    :param fac_ret_cap: MultiIndex pd.DataFrame
    :param col_name: str
        根据该列值的大小进行排序
    :param num_group: int
        分成的组数
    :param ascending: Bool
        True: ascending, False: Descending

    :return pd.DataFrame
        在原先的data后增加一列，每个元素对应的是该行对应的组数
    """
    dfg = fac_ret_cap.groupby(level=0).apply(lambda frame: cut_group(frame, col_name, num_group, ascending))
    return dfg



