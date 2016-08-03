# coding=utf-8
from __future__ import division, unicode_literals, print_function

import itertools
import math
from collections import deque, Counter
from functools import partial
from operator import itemgetter
import statsmodels.api as sm
import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr, pearsonr

from alpha.alpha_algo.const import FREQ_NUM
from alpha.util import timethis
from util import (down_side_stdev, max_drawdown, ICAnalysis,
                  ReturnAnalysis, FactorData, TurnOverAnalysis,
                  IndustryAnalysis, get_stock_industry, CodeAnalysis)


# def _normal_IC(df_fac, df_ret):
#     """
#     信息系数计算
#     """
#     ic, pvalue = pearsonr(df_fac, df_ret)
#
#     return ic, pvalue
#
#
# def _rank_IC(df_fac, df_ret):
#     """
#     排序信息系数
#     """
#     ret = spearmanr(df_fac, df_ret)
#     return list(ret)
#
#
# def _risk_IC(df_fac, df_ret, cov):
#     """
#     风险调整信息系数
#     cov协方差矩阵
#     TODO: check error
#     """
#     n = len(df_fac)
#     W = np.ones([n]) / n
#     rf = 0.02
#     R = df_ret.values
#     target = lambda W: 1 / \
#                        ((sum(W * R) - rf) / math.sqrt(
#                            np.dot(np.dot(W, cov), W)))
#     b = [(0., 1.) for i in range(n)]  # boundary condition
#     c = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # summation condition
#     optimized = scipy.optimize.minimize(
#         target, W, method='SLSQP', bounds=b, constraints=c)
#     weights = optimized.x
#     df_ret_w = df_ret * weights
#     ret = pearsonr(df_fac, df_ret_w)
#     return list(ret)
#
#
# def IC(_df_fac, _df_ret, cov=None, method='normal'):
#     if _df_ret is None:
#         return (np.nan, np.nan)
#     df_fac = _df_fac.dropna() if _df_fac.isnull().any() else _df_fac
#     df_ret = _df_ret.dropna() if _df_ret.isnull().any() else _df_ret
#
#     ret1, ret2 = df_fac.align(df_ret, join='inner')
#     if method == 'normal':
#         return _normal_IC(ret1, ret2)
#     elif method == 'rank':
#         return _rank_IC(ret1, ret2)
#     elif method == 'risk':
#         return _risk_IC(ret1, ret2, cov)


def group_IC(df_fac, df_ret, method='normal'):
    """
    由于是分组IC，不需要考虑index对齐问题
    Args:
        df_fac:
        df_ret:
        method:

    Returns:

    """
    if method == 'normal':
        ic = _normal_IC(df_fac, df_ret)
        return ic
    elif method == 'rank':
        return _rank_IC(df_fac, df_ret)


def cut_group(df, factor_name, num_group, ascending):
    """
    根据因子值排序，平均分割df 为num_group组
    Args:
        df:
        factor_name:
        num_group:
        ascending:

    Returns: series, index:dt, 元素：列表,['Q2','Q5', ...],表明每个secu对应的分组

    """
    N = len(df)
    # 平均每组元素个数
    avg_element = N // num_group
    remains = N % num_group
    each_group = [avg_element] * num_group
    # 把剩下的平均分配到前面各组
    if remains:
        for idx in range(0, remains):
            each_group[idx] += 1
    each_group = np.array(each_group)
    each_group = each_group.cumsum()
    idx = None
    try:
        idx = df.loc[:, factor_name].rank(method='first', na_option='bottom', ascending=ascending)
    except:
        print(idx)
        print(df.index[0])
        print(factor_name)
        print('error occurred in cut_group')
    idx = pd.Series(idx)
    groups = pd.Series([''] * N, index=idx.index)
    start = 0
    for grp, end in enumerate(each_group):
        mask = (idx > start) & (idx <= end)
        groups[mask] = ''.join(['Q', str(grp + 1)])
        start = end
    groups = groups.tolist()

    return groups


def _phil_group_ret_mean(_data, factor_name, fac_ret_cap=None, method='eqwt'):
    # factor_name = list(set(data.columns) - {'codes', 'ret', 'cap'})[0]
    dt = _data.index[0][0]
    if fac_ret_cap.xs(dt)[factor_name].isnull().all():
        return np.nan
    else:
        data = _data.copy()
        ret_mean = np.nan
        if method == 'eqwt':
            ret_mean = data.ret.mean()
        elif method == 'capwt':
            ret_mean = (data.ret * data.cap).sum() / data.cap.sum()
        elif method == 'fv':
            ret_mean = (data.ret * data.loc[:, factor_name]
                        ).sum() / data.loc[:, factor_name].sum()
        elif method == 'f_score':
            argsort_value = data.loc[:, factor_name].argsort()
            ret_mean = (data.ret * argsort_value).sum() / argsort_value.sum()
        return ret_mean


def _phil_group_ic(_data, factor_name, method='normal'):
    # factor_name = list(set(data.columns) - {'codes', 'ret', 'cap'})[0]
    data = _data[[factor_name, 'ret', 'cap']]
    if data.isnull().any().any():
        data_no_na = data.dropna()
        fac = data_no_na.loc[:, factor_name]
        ret = data_no_na.loc[:, 'ret']
    else:
        fac = data.loc[:, factor_name]
        ret = data.loc[:, 'ret']

    ic, pvalue = group_IC(fac, ret, method=method)
    return ic, pvalue


def _phil_ex_num_ratio(data, bench_ret_term):
    dt = data.index.values[0][0]  # [('2012-01-01', '000001'),...]
    benchmark_ret = bench_ret_term.ix[dt, 0]
    ret = ((data.ret.dropna() - benchmark_ret) > 0).mean() * 100
    return ret


def phil_return_analysis(factor_name, grouped, freq, benchmark_term_return,
                         num_group, ascending, return_mean_method, g_sell):
    """
        对分组收益率进行相关统计及分析，指标有：
        每组平均收益率：r_mean
        每组累计收益率: r_cum
        年化收益率：r_cum_y
        最大盈利：r_max
        最小盈利: r_min
        最大回撤：maxd
        标准差：sigm
        年化变准差：sigma_y
        夏普比率：sharp_ratio

        分组收益率与对照指数比较的统计指标:
        累计超额收益: ex_r_cum
        年化超额收益：ex_r_anual
        最大超额收益：ex_r_max
        最小超额收益：ex_r_min
        beta, alpha
        收益超过bench的股票个数：ex_num
        胜率: win_ratio
        信息比率：超额收益率均值/超额收益率年化标准差

    Args:
        return_mean_method:
        num_group:
        ascending:
        grouped:
        factor_name:
        freq:
        """
    # +++++分组收益率分析+++++
    # 分组每个调仓期的收益率
    # df_group = self._group_fac_data(df_com_fac)
    # ret_mean = self._group_ret_mean(
    #     df_group, method=ret_mean_method)
    if hasattr(grouped, 'groups'):
        grouped = grouped
    else:
        grouped = get_grouped_data(fac_ret_cap=grouped,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)

    _fac_ret_cap = grouped.apply(lambda x: x)

    group_ret_mean = partial(_phil_group_ret_mean,
                             fac_ret_cap=_fac_ret_cap,
                             factor_name=factor_name,
                             method=return_mean_method)
    group_ret_mean.__module__ = _phil_group_ret_mean.__module__
    group_ret_mean.__name__ = _phil_group_ret_mean.__name__
    ex_num_ratio = partial(_phil_ex_num_ratio,
                           bench_ret_term=benchmark_term_return)
    ex_num_ratio.__module__ = _phil_ex_num_ratio.__module__
    ex_num_ratio.__name__ = _phil_ex_num_ratio.__name__
    ret_mean = grouped.apply(group_ret_mean).unstack()
    try:
        ret_mean.loc[:, 'Q_LS'] = ret_mean[unicode('Q1')] - ret_mean[unicode(g_sell)]
    except KeyError as e:
        print(e)
    ret_mean = ret_mean.T

    df_stats = pd.DataFrame()
    # 分组每个调仓期的累计收益率
    g_ret_cum = (ret_mean + 1).cumprod(axis=1) - 1
    # 平均收益率
    df_stats['r_mean'] = ret_mean.mean(axis=1)
    # 全部时间的累计收益率
    df_stats['r_cum'] = g_ret_cum.ix[:, -1]
    y_dic = FREQ_NUM
    y_num = y_dic[freq]
    y_factor = y_num / len(ret_mean.columns)
    # 累计收益率(年化)
    df_stats['r_cum_y'] = (
                              1 + df_stats['r_cum']) ** y_factor - 1
    # 回测区间最大收益率
    df_stats['r_min'] = ret_mean.min(axis=1)
    # 回测区间最小收益率
    df_stats['r_max'] = ret_mean.max(axis=1)
    # 标准差
    sigma = ret_mean.std(axis=1)
    # 标准差（年化）
    df_stats['sigma_y'] = sigma * np.sqrt(y_num)
    # 下行标准差（年化）
    df_stats['d_sigma'] = ret_mean.apply(
        lambda x: down_side_stdev(x) * np.sqrt(y_num), axis=1)
    # 最大回撤
    df_stats['max_drawdown'] = g_ret_cum.apply(
        lambda x: max_drawdown(x + 1), axis=1)
    # 夏普比率
    df_stats['sharp_ratio'] = df_stats['r_cum_y'] / df_stats['sigma_y']

    # +++++分组收益率与bench_mark作比较分析+++++
    # bench_mark 收益率
    # benchmark_term_return = self.all_data['benchmark_term_return']
    if isinstance(benchmark_term_return, pd.DataFrame):
        benchmark_term_return = benchmark_term_return.ix[:, 0]
    # bench_mark 累计收益率
    bench_r_cum = (benchmark_term_return + 1).cumprod() - 1
    # bench_mark 累计收益率(年化)
    bench_r_cum_y = (1 + bench_r_cum[-1]) ** y_factor - 1
    # 分组个股收益率
    df_stats['ex_num_ratio'] = grouped.apply(ex_num_ratio).unstack().mean()
    # 分组每期超额收益率
    g_ex_ret = ret_mean - benchmark_term_return.T
    # 最大超额收益率
    df_stats['ex_r_max'] = g_ex_ret.max(axis=1)
    # 最小超额收益率
    df_stats['ex_r_min'] = g_ex_ret.min(axis=1)
    # 跟踪误差（年化）
    df_stats['track_error'] = g_ex_ret.std(axis=1) * np.sqrt(y_num)
    # 超额收益率(年化)
    df_stats['ex_r_cum_y'] = df_stats['r_cum_y'] - bench_r_cum_y
    # 信息比率
    df_stats['info_ratio'] = df_stats['ex_r_cum_y'] / df_stats['track_error']
    # 胜率
    df_stats['win_ratio'] = g_ex_ret.apply(
        lambda x: sum(x > 0) / float(len(x)), axis=1)
    # alpha & beta
    x = sm.add_constant(benchmark_term_return)
    # for col in ret_mean.T.columns:
    #     try:
    #         print(sm.OLS(ret_mean.T[col].dropna().values, x.loc[ret_mean.T[col].dropna().index, :].values).fit().params[
    #                   0])
    #     except:
    #         pass
    df_stats['alpha'] = [
        sm.OLS(ret_mean.T[col].dropna().values, x.loc[ret_mean.T[col].dropna().index, :].values).fit().params[0] for col
        in ret_mean.T.columns]
    df_stats['beta'] = [
        sm.OLS(ret_mean.T[col].dropna().values, x.loc[ret_mean.T[col].dropna().index, :].values).fit().params[1] for col
        in ret_mean.T.columns]
    # 返回收益率分析指标：df_stats, 分组平均收益率：ret_mean, 分组累计收益率：
    # g_ret_cum， 分组IC：
    columns = sorted(df_stats.columns)
    df_stats = df_stats.loc[:, columns]
    ret = ReturnAnalysis(name=factor_name, return_statistics=df_stats,
                         group_return_mean=ret_mean,
                         group_return_cumulative=g_ret_cum)
    return ret


def get_grouped_data(fac_ret_cap, factor_name, num_group, ascending):
    __cut_group = lambda df: cut_group(df, factor_name, num_group, ascending)
    group_names = fac_ret_cap.groupby(level=0).apply(__cut_group).sum()
    grouped = fac_ret_cap.groupby([pd.Grouper(level=0), group_names])
    return grouped


@timethis
def phil_ic_analysis(factor_name, grouped, fac_ret_cap, num_group, ascending,
                     ic_method):
    #  分组IC, num_group = numgroup
    if grouped is None:
        grouped = get_grouped_data(fac_ret_cap=fac_ret_cap,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)

    __group_IC = lambda df: _phil_group_ic(df, factor_name=factor_name,
                                           method=ic_method)
    _group_ic = grouped.apply(__group_IC).unstack()
    group_ic_ic = _group_ic.applymap(itemgetter(0))
    group_ic_p_value = _group_ic.applymap(itemgetter(1))
    group_ic = pd.Panel({'ic': group_ic_ic, 'p_value': group_ic_p_value})

    # 不分组IC, num_group=1
    non_group_ic = fac_ret_cap.groupby(level=0).apply(__group_IC)
    ic = pd.DataFrame({'ic': non_group_ic.map(itemgetter(0)),
                       'p_value': non_group_ic.map(itemgetter(1))})

    ic_statistics = {'IC_mean': ic.ic.mean(), 'p_mean': ic.p_value.mean(),
                     'IC_Stdev': ic.ic.std(),
                     'IC_IR': ic.ic.mean() / ic.ic.std()}
    # ic_decay
    ic_decay = _phil_IC_decay(fac_ret_cap, factor_name)

    ic_analysis = ICAnalysis(IC_series=ic, IC_statistics=ic_statistics,
                             groupIC=group_ic, IC_decay=ic_decay)
    return ic_analysis


def __calc_percent(dic):
    """

    Args:
        dic: Counter obj
    Returns:
    """
    total = sum(dic.values())
    ret = {k: v / total for k, v in dic.iteritems()}
    return ret


@timethis
def phil_industry_analysis(factor_name, grouped, num_group, all_stock_ind,
                           ascending=False):
    if hasattr(grouped, 'groups'):
        groups = grouped
    else:
        groups = get_grouped_data(grouped, factor_name, num_group,
                                  ascending=ascending)

    func = partial(__code_industry_map_and_count, all_stock_ind=all_stock_ind)
    func.__name__ = __code_industry_map_and_count.__name__
    func.__module__ = __code_industry_map_and_count.__module__
    df_counts = groups.apply(func).unstack()
    df_counts_sum = df_counts.sum()  # series, index is group_name
    df_counts_sum = pd.DataFrame(list(df_counts_sum.values),
                                 index=df_counts_sum.index)
    gp_mean_per = df_counts_sum.T / df_counts_sum.T.sum()
    df_percents = df_counts.applymap(__calc_percent)

    industry_analysis = IndustryAnalysis(gp_mean_per=gp_mean_per,
                                         gp_industry_percent=df_percents)

    return industry_analysis


def __code_industry_map_and_count(df, all_stock_ind):
    return Counter(all_stock_ind.loc[df.dropna().reset_index().secu, 'ind'])


@timethis
def phil_single_factor_analysis(factor_name, fac_ret_cap, freq, g_sell,
                                benchmark_term_return, turnover_method='count',
                                ic_method='normal', return_mean_method='eqwt',
                                num_group=5, fp_month=12, g_buy='Q1',
                                sam_level=1, all_stocks=None,
                                ascending=None):
    if hasattr(fac_ret_cap, 'groups'):
        grouped = fac_ret_cap
    else:
        grouped = get_grouped_data(fac_ret_cap=fac_ret_cap,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)
    # return analysis
    return_analysis = phil_return_analysis(factor_name=factor_name,
                                           grouped=grouped, freq=freq,
                                           benchmark_term_return=benchmark_term_return,
                                           num_group=num_group,
                                           ascending=ascending,
                                           return_mean_method=return_mean_method,
                                           g_sell=g_sell)
    # IC
    ic_analysis = phil_ic_analysis(factor_name=factor_name, grouped=grouped,
                                   fac_ret_cap=fac_ret_cap,
                                   num_group=num_group, ascending=ascending,
                                   ic_method=ic_method)

    # turnover analysis
    turnover_analysis = phil_turnover_analysis(factor_name=factor_name,
                                               grouped=grouped,
                                               fac_ret_cap=fac_ret_cap,
                                               fp_month=fp_month, g_buy=g_buy,
                                               g_sell=g_sell,
                                               num_group=num_group,
                                               turnover_method=turnover_method,
                                               ascending=ascending)

    # code_analysis
    code_analysis = phil_code_analysis(factor_name, grouped, all_stocks,
                                       num_group, sam_level, ascending)
    factor_data = FactorData(name=factor_name,
                             return_analysis=return_analysis,
                             IC_analysis=ic_analysis,
                             turnover_analysis=turnover_analysis,
                             code_analysis=code_analysis)

    return factor_data


def __cap_mean(df):
    return df.dropna()['cap'].mean()


def phil_code_analysis(factor_name, grouped, all_stocks, num_group, sam_level,
                       ascending):
    if not hasattr(grouped, 'groups'):
        groups = get_grouped_data(factor_name=factor_name,
                                  fac_ret_cap=grouped,
                                  num_group=num_group,
                                  ascending=ascending)
    else:
        groups = grouped
    all_stock_ind = get_stock_industry(all_stocks, sam_level=sam_level)
    industry_analysis = phil_industry_analysis(factor_name, groups, num_group,
                                               all_stock_ind)
    # cap
    cap = grouped.apply(__cap_mean).unstack().T
    cap.loc[:, 'group_cap_mean'] = cap.mean(axis=1)
    # group stocks
    stocks = dict(list(groups))
    for (dt, g), v in stocks.iteritems():
        stocks[(dt, g)] = v.index.get_level_values(level=1).tolist()
    code_analysis = CodeAnalysis(industry_analysis=industry_analysis,
                                 cap_analysis=cap,
                                 stock_list=stocks)
    return code_analysis


@timethis
def phil_turnover_analysis(factor_name, grouped, fac_ret_cap, fp_month, g_buy,
                           g_sell, num_group, turnover_method,
                           ascending):
    if grouped is None:
        grouped = get_grouped_data(fac_ret_cap=fac_ret_cap,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)
    turnover = _phil_turnover(factor_name=factor_name, grouped=grouped,
                              method=turnover_method, num_group=num_group,
                              ascending=ascending)
    auto_correlation = _phil_autocorr(factor_name, fac_ret_cap)
    buy_signal = _phil_buy_signal(factor_name=factor_name, grouped=grouped,
                                  fp_month=fp_month, num_group=num_group,
                                  g_buy=g_buy, g_sell=g_sell,
                                  ascending=ascending)
    turnover_analysis = TurnOverAnalysis(buy_signal=buy_signal,
                                         auto_correlation=auto_correlation,
                                         turnover=turnover)
    return turnover_analysis


# @timethis
def _phil_IC_decay(fac_ret_cap, factor_name):
    grouped = fac_ret_cap.groupby(level=0)
    n = len(grouped)
    lag = min(n, 12)

    rets = []
    dts = [dt for dt, _ in grouped]
    frames = (frame.reset_index(level=0, drop=True) for _, frame in grouped)
    for piece_data in window(frames, lag, longest=True):
        ret = [IC(df_fac.loc[:, factor_name], df_ret.loc[:, 'ret'])[0]
               if df_ret is not None else np.nan
               for df_fac, df_ret in zip([piece_data[0]] * lag, piece_data)]
        rets.append(ret)

    columns = [''.join(['lag', str(i)]) for i in range(lag)]
    df = pd.DataFrame(rets, index=dts[:len(rets)], columns=columns)
    decay = df.mean().to_frame()
    decay.columns = ['decay']
    return decay


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


def __count_turnover(current_df, next_df):
    current_codes = set(current_df.secu)
    next_codes = set(next_df.secu)
    if len(current_codes) != 0:
        ret = len((next_codes - current_codes)) / len(current_codes)
    elif len(next_codes) != 0:
        ret = 0.0
    else:
        ret = np.nan
    return ret


def __capwt_turnover(current_df, next_df):
    current_weights = current_df.cap / current_df.cap.sum()
    next_weights = next_df.cap / next_df.cap.sum()

    cur, nxt = current_weights.align(next_weights, join='outer', fill_value=0)
    ret = (cur - nxt).abs().sum() / 2
    return ret


def __get_secus_and_caps(df):
    from collections import namedtuple
    SecuCap = namedtuple('SecuCap', ['secu', 'cap'])
    flat_df = df.dropna().reset_index()
    return SecuCap(flat_df.secu, flat_df.cap)


@timethis
def _phil_turnover(factor_name, grouped, method, num_group=5, ascending=True):
    if hasattr(grouped, 'groups'):
        grouped = grouped
    else:
        grouped = get_grouped_data(fac_ret_cap=grouped,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)
    frame = grouped.apply(__get_secus_and_caps).unstack()
    valid_methods = {
        'count': __count_turnover,
        'capwt': __capwt_turnover,

    }
    ret = []
    for idx in range(0, len(frame) - 1):
        subframe = frame.iloc[idx:(idx + 2), :]
        ret.append([valid_methods[method](*subframe.loc[:, col]) for col in
                    subframe.columns])
    columns = frame.columns
    idx = frame.index[1:len(ret) + 1]
    df = pd.DataFrame(ret, index=idx, columns=columns)
    return df


# @timethis
def __buy_signal_or_reversal(current, next_):
    # current_dt, group, current_df = current
    # next_dt, _, next_df = next_
    current_codes = set(current.secu)
    next_codes = set(next_.secu)
    ret = len(current_codes.intersection(next_codes)) / len(current_codes) if len(current_codes) != 0 else np.nan
    return ret


@timethis
def _phil_buy_signal(factor_name, grouped, fp_month=12, num_group=5,
                     g_buy='Q1', g_sell='Q5', ascending=True):
    if not hasattr(grouped, 'groups'):
        grouped = get_grouped_data(fac_ret_cap=grouped,
                                   factor_name=factor_name,
                                   num_group=num_group, ascending=ascending)
    n = len(grouped)
    lag = min(n, fp_month)
    frame = grouped.apply(__get_secus_and_caps).unstack()
    group_buy = deque(frame.loc[:, g_buy], maxlen=lag)
    try:
        group_sell = deque(frame.loc[:, g_sell], maxlen=lag)
    except:
        pass
    decay = [__buy_signal_or_reversal(current, next_)
             for current, next_ in zip([group_buy[0]] * lag, group_buy)]
    reversal = [__buy_signal_or_reversal(current, next_)
                for current, next_ in zip([group_buy[0]] * lag, group_sell)]
    dts = frame.index[-12:]
    decay = pd.Series(decay, index=dts, name='decay')
    reversal = pd.Series(reversal, index=dts, name='reversal')
    ret = pd.concat([decay, reversal], axis=1)
    return ret


# @timethis
def __autocorr(df1, df2, factor_name):
    if df2 is None:
        return np.nan

    fac1 = df1.loc[:, factor_name]
    fac2 = df2.loc[:, factor_name]

    if fac1.isnull().any():
        fac1.dropna(inplace=True)
    if fac2.isnull().any():
        fac2.dropna(inplace=True)

    ret1, ret2 = fac1.align(fac2, join='inner')
    rho, pvalue = pearsonr(ret1, ret2)

    return rho


@timethis
def _phil_autocorr(factor_name, fac_ret_cap):
    grouped = fac_ret_cap.groupby(level=0)
    n = len(grouped)
    lag = min(n, 12)
    index = [dt for dt, _ in grouped]
    frames = (frame.reset_index(drop=True, level=0) for _, frame in grouped)
    columns = [''.join(['lag', str(i)]) for i in range(lag)]

    results = []
    for data in window(frames, lag, longest=True):
        # data = map(itemgetter(1), data)  # data is ((dt, frame), ...), get the frame
        ret = [__autocorr(df1, df2, factor_name) if df2 is not None else np.nan
               for df1, df2 in zip([data[0]] * lag, data)]
        results.append(ret)
    auto_correlation = pd.DataFrame(results, index=index, columns=columns)
    return auto_correlation


if __name__ == '__main__':
    df = pd.DataFrame([1, 3, 5, 9, 2, 8, 7], columns=['A'])
    grp = cut_group(df, 'A', 3, True)
    print(grp)
    print('done!')
