# -*- coding: utf-8 -*-

import itertools

import numpy as np
from joblib import Parallel, delayed
from six import string_types

from .data_type import *
from .get_data import *
from .metrics import return_perf_metrics, information_coefficient


def prepare_data(factor_name, index_code, start_date, end_date, freq):
    """
    获取因子数据及股票对应的下一期收益率与市值
    Args:
        factor_name (str): 因子名称, 例如 'M004023'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: 原始因子与其下期收益率
    """
    if not isinstance(factor_name, string_types):
        raise TypeError('factor name should be a string, but its type is {}'.format(type(factor_name)))
    factor_names = [factor_name] + ['M004023']
    factor_names = [str(n) for n in factor_names]
    raw_fac = get_raw_factor(factor_names, index_code, start_date, end_date, freq)

    dts = sorted(raw_fac.index.get_level_values(0).unique())
    s, e = str(dts[0]), str(dts[-1])

    stocks = sorted([str(c) for c in raw_fac.index.get_level_values(1).unique()])

    close_price = Parallel(n_jobs=10, backend='threading', verbose=5)(
            delayed(csf.get_stock_hist_bar)(code, freq,
                                            start_date=s,
                                            end_date=e,
                                            field=['date', 'close'])
            for code in stocks)
    for s, p in zip(stocks, close_price):
        p['tick'] = s
    close_price = pd.concat(close_price)

    close_price = close_price.dropna()

    # index.name原来为空
    close_price.index.name = 'dt'

    # 转成一个frame, index:dt, columns:tick
    close_price = (close_price.set_index('tick', append=True)
                   .to_panel()['close']
                   .sort_index()
                   .fillna(method='ffill')
                   )
    # 取每个周期末
    group_key = {'M': [close_price.index.year, close_price.index.month],
                 'W': [close_price.index.year, close_price.index.week],
                 'Q': [close_price.index.year, close_price.index.quarter]
                 }
    close_price = close_price.groupby(group_key[freq]).tail(1)

    returns = close_price.pct_change().shift(-1).dropna(axis=1, how='all')

    returns.index = returns.index.map(lambda dt: str(dt.date()))
    returns.index.name = 'dt'
    returns = returns.unstack().to_frame()
    returns.columns = ['ret']
    returns = returns.swaplevel(0, 1).sort_index()
    returns.index.names = raw_fac.index.names

    # 去掉最后一期数据
    # 去掉由于停牌等无法算出收益率的股票
    fac_ret = raw_fac.join(returns).dropna()

    return fac_ret


def add_group(fac_ret, num_group=5):
    """
    添加一个分组列
    Args:
        num_group (int): 组数
        fac_ret (DataFrame): 一个Multi-index数据框, 含有因子,市值, 下期收益率数据
    """

    def __add_group(frame):

        keep_columns = ['M004023', 'ret']
        factor_name = list(set(frame.columns) - set(keep_columns))
        factor_name = factor_name[0]
        rnk = frame.loc[:, factor_name].rank(method='first', na_option='bottom')
        labels = ['Q{:0>2}'.format(i) for i in range(1, num_group + 1)]
        category = pd.cut(rnk, bins=num_group, labels=labels).astype(str)
        category.name = 'group'
        new_frame = frame.join(category)
        return new_frame

    if 'group' in fac_ret.columns:
        return fac_ret
    else:
        return fac_ret.groupby(level=0).apply(__add_group)


def raw_data_plotting():
    pass


def return_analysis(fac_ret_data, bench_returns, fac_name=None, plot=False):
    """
    收益率分析
    Args:
    :fac_ret_data (DataFrame): 含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'
    :param bench_returns:
    :param fac_name:
    :param plot:
    :return:
    """
    group_mean = fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data['group']])[['ret']].mean()
    group_mean = group_mean.to_panel()['ret']
    group_mean['Q_LS'] = group_mean.ix[:, 0] - group_mean.ix[:, -1]
    return_stats = pd.DataFrame()
    for col in group_mean.columns:
        return_stats[col] = return_perf_metrics(group_mean[col], bench_returns)

    ret = ReturnAnalysis()
    ret.return_stats = return_stats
    ret.group_mean_return = group_mean
    ret.group_cum_return = (group_mean + 1).cumprod() - 1
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

    ic_statistics = pd.Series({'IC_mean' : ic_series.ic.mean(), 'p_mean': ic_series.p_value.mean(),
                               'IC_Stdev': ic_series.ic.std(),
                               'IC_IR'   : ic_series.ic.mean() / ic_series.ic.std()})

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
