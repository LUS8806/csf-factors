# -*- coding: utf-8 -*-
import itertools

import numpy as np

from config import STOCK_FILTER
from get_data import *


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
        data_ = data.to_frame()
    data_ = data.copy()
    factor_name = get_factor_name(data)
    mu = data[factor_name].mean()
    ind = mean_abs_deviation(data[factor_name]) if method == 'mad' else data[factor_name].std()
    try:
        data_.loc[:, factor_name] = data_[factor_name].clip(lower=mu - num * ind, upper=mu + num * ind)
    except Exception, e:
        print(e.message)
    return data_


def data_scale(data, method='normal'):
    """
    数据标准化处理
    Args:
        data (DataFrame): 数据框
        method (str): 标准化方法
        "normal": (x-x.mean())/x.std()
        "cap": (x - cap_weighted_mean of x)/x.std()
    Returns:
        DataFrame 标准化处理后的数据
    Raises:
        ValueError
    """
    data_ = data.copy()

    factor_name = get_factor_name(data_)

    if method == 'normal':
        data_.loc[:, factor_name] = ((data_.loc[:, factor_name] - data_.loc[:, factor_name].mean())
                                     / data_.loc[:, factor_name].std())
    elif method == 'cap':
        cap_weight = data_.loc[:, 'cap'] / data_.loc[:, 'cap'].sum()
        avg = (data_.loc[:, 'cap'] * cap_weight).sum()
        data_.loc[:, factor_name] = (data_.loc[:, factor_name] -avg) / data_.loc[:, factor_name].std()
    else:
        raise ValueError('标准化算法现在仅支持normal与cap')

    return data_


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


def get_factor_name(fac_ret):
    """
    在单因子分析中,有因子,因子对应的下期收益率, benchmark下期收益率, 市值, 分组列,本函数取得因子列的列名称.

    Args:
        fac_ret: 一个数据框

    Returns:
        str: 因子名称
    """
    keep_columns = ['cap', 'benchmark_returns', 'ret', 'group']
    factor_name = set(fac_ret.columns) - set(keep_columns)
    # assert len(factor_name) == 1, "there should be only one factor, got {}".format(factor_name)
    factor_name = factor_name.pop()
    return factor_name


def window(seq, n=2, longest=False):
    """Returns a sliding window (of width n) over data from the iterable
       s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...

    Args:
        longest: if True, get full length of seq,
        e.g. window([1,2,3,4], 3, longest=True) --->
        (1,2,3), (2,3,4), (3,4,None), (4,None,None)
    """
    if longest:
        it = itertools.chain(iter(seq), [None] * (n - 1))
    else:
        it = iter(seq)
    result = tuple(itertools.islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def __frame_to_rank(direction, standardized_factor):
    '''
    根据因子方向调整正负。
    '''
    standardized_factor_copy = standardized_factor.copy()
    # transfer True to -1, False to 1
    asc = {}
    for k, v in direction.iteritems():
        new_value = -1 if v else 1
        asc[k] = new_value
    # *(-1) if ascending=True
    for col in standardized_factor_copy.columns:
        if col not in ['cap']:
            standardized_factor_copy[col]=standardized_factor[col]*asc[col]
    return standardized_factor_copy

def form_fenceng(fenceng, fac_codes):
    '''
    根据ic计算各情景因子权重矩阵..
    '''
    df = pd.read_csv('/media/jessica/00001D050000386C/SUNJINGJING/dev/csf-factors/factors/stats_M004023.csv')
    lst = list(set(fac_codes)|set(['dt', 'Unnamed: 0']))
    df = df[df['Unnamed: 0'] != 'all']
    df = df[lst]
    df['Unnamed: 0']=df['Unnamed: 0'].map(lambda x:('{}_{}').format('cap',x))
    df = df.set_index(['dt', 'Unnamed: 0']).sort_index()
    ret=df.sort_index(level=0)
    ret = ret.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).T)
    #############     #############    #############    #############    #############    计算IC_IR,并根据icir进行加权
    ic=ret.swaplevel(0, 1).sort_index()
    ir={}
    for ix in fac_codes:
        ir[ix]=ic.xs(ix).rolling(12,min_periods=1).mean()/ic.xs(ix).rolling(12,min_periods=1).std()
    ret=pd.Panel(ir).to_frame(filter_observations=False)
    ret = ret.groupby(level=0).apply(lambda x: x.reset_index(level=0, drop=True).T)
    ##    #############    #############    #############    #############    #############    ############# end
    ret = ret.abs().fillna(0)  # na用o填充，全部变为绝对值
    ret=ret.groupby(level=0).apply(lambda x: x / x.sum())  # 计算百分比
    ret=ret.fillna(0.0)
    return ret

def drop_middle(df):
    import numpy as np
    x=df.rank(method='average', na_option='keep')
    l=len(x)
    high=('{}_high').format('cap')
    low = ('{}_low').format('cap')
    if x.loc[:, 'cap'].notnull().sum()==0:
        x.loc[:, high]=np.nan
        x.loc[:, low]=np.nan
    else:
        x.loc[:, high] =pd.cut(x.loc[:, 'cap'], bins=9, labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]).values
        x.loc[:, low] = pd.cut(x.loc[:, 'cap'], bins=9, labels=[-9, -8, -7, -6, -5, -4, -3, -2, -1]).values
    x=x.drop('cap',1)
    # x=x.applymap(lambda y: abs(y) if isinstance(y,int) else 0.0)
    return x.abs()

def fenceng_score(df):
    '''
    df3:对于不同个股，按照披露值进行打分，并得到个股情景权重
    '''
    if not isinstance(df,pd.DataFrame):
        df=pd.DataFrame(df)
    ret = df.groupby(level=0).rank(method='average', na_option='keep')
    __drop_middle = lambda x: drop_middle(x)
    ret2 = ret.groupby(level=0).apply(__drop_middle).sort_index()
    ret2=ret2.fillna(0.0)
    df3=ret2.groupby(level=0).apply(lambda x:x.reset_index(level=0, drop=True).T)
    df3 = df3.sort_index().groupby(level=0).apply(lambda x: x / x.sum())  # 计算百分比
    return df3

