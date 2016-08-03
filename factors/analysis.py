# -*- coding: utf-8 -*-

from get_data import *
from metrics import return_perf_metrics, information_coefficient
from data_type import *
import itertools
import numpy as np


def prepare_data(fac_names, index_code, start_date, end_date, freq):
    """
    获取因子数据及股票对应的下一期收益率与市值

    :param index:
    :param fac_names:
    :param start_date:
    :param end_date:
    :param freq:
    :return:
    """
    raw_fac_names = fac_names
    if not isinstance(fac_names, list):
        fac_names = fac_names
    fac_names.append('M004023')
    raw_fac = get_raw_factor(fac_names, index_code, start_date, end_date, freq)

    dt_index = raw_fac.index.get_level_values(0).unique()

    all_stocks = raw_fac.index.get_level_values(1).unique()

    pass


def raw_data_plotting():

    pass


def return_analysis(fac_ret_data, bench_returns, fac_name=None, plot=False):
    """
    收益率分析
    :param fac_ret_data:
    :param bench_returns:
    :param fac_name:
    :param plot:
    :return:
    """
    group_mean = fac_ret_data.groupby(level=0).apply(lambda frame: frame.groupby('group')['ret'].mean())
    group_mean['Q_LS'] = group_mean.ix[:, 0] - group_mean.ix[:, -1]
    return_stats = pd.DataFrame()
    for col in group_mean.columns:
        return_stats[col] = return_perf_metrics(group_mean[col], bench_returns)

    ret = ReturnAnalysis()
    ret.return_stats = return_stats
    ret.group_mean_return = group_mean
    ret.group_cum_return = (group_mean+1).cumprod()-1
    return ret


def information_coefficient_analysis(fac_ret_data, plot=False, ic_method='normal'):
    """
    信息系数（IC）分析
    :param fac_ret_data:
    :param plot:
    :param ic_method
    :return:
    """
    ic_series = fac_ret_data.groupby(level=0).apply(
        lambda frame: information_coefficient(frame['factor'], frame['ret'], ic_method))

    ic_decay = IC_decay(fac_ret_data)

    group_ic = fac_ret_data.groupby(level=0).apply(lambda frame: frame.groupby('group').apply(
        lambda df: information_coefficient(df['factor'], df['ret'], ic_method)))

    ic_statistics = pd.Series({'IC_mean': ic_series.ic.mean(), 'p_mean': ic_series.p_value.mean(),
                     'IC_Stdev': ic_series.ic.std(),
                     'IC_IR': ic_series.ic.mean() / ic_series.ic.std()})

    ret = ICAnalysis()
    ret.IC_series = ic_series
    ret.IC_decay = ic_decay
    ret.IC_statistics = ic_statistics
    ret.groupIC = group_ic
    return ret


def IC_decay(fac_ret_cap):
    """
    信息系数衰减
    :param fac_ret_cap:
    :return:
    """
    grouped = fac_ret_cap.groupby(level=0).apply(lambda frame: frame.groupby('group'))
    n = len(grouped)
    lag = min(n, 12)

    rets = []
    dts = [dt for dt, _ in grouped]
    frames = (frame.reset_index(level=0, drop=True) for _, frame in grouped)
    for piece_data in window(frames, lag, longest=True):
        ret = [information_coefficient(df_fac.loc[:, 'factor'], df_ret.loc[:, 'ret'])[0]
               if df_ret is not None else np.nan
               for df_fac, df_ret in zip([piece_data[0]] * lag, piece_data)]
        rets.append(ret)

    columns = [''.join(['lag', str(i)]) for i in range(lag)]
    df = pd.DataFrame(rets, index=dts[:len(rets)], columns=columns)
    decay = df.mean().to_frame()
    decay.columns = ['decay']
    return decay


def turnover_analysis(fac_ret_data, plot=False):
    """
    换手率分析
    :param fac_ret_data:
    :param plot:
    :return:
    """
    ret = TurnOverAnalysis()
    return ret


def code_analysis(fac_ret_data, plot=False):
    """
    选股结果分析
    :param fac_ret_data:
    :param plot:
    :return:
    """
    ret = CodeAnalysis()
    return ret


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




