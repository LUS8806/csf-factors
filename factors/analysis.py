# -*- coding: utf-8 -*-

import itertools
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from six import string_types

from data_type import *
from factors.util import get_factor_name
from get_data import *
from metrics import return_perf_metrics, information_coefficient
from util import data_scale
from util import extreme_process


def prepare_data(factor_name, index_code, benchmark_code, start_date, end_date, freq):
    """
    获取因子数据及股票对应的下一期收益率与市值
    Args:
        benchmark_code:
        factor_name (str): 因子名称, 例如 'M004023'
        index_code (str): 六位指数代码, 例如 '000300'
        start_date (str): 开始日期, YYYY-mm-dd
        end_date (str): 结束日期, YYYY-mm-dd
        freq (str): 数据频率, m=month
    Returns:
        DataFrame: 原始因子与其下期收益率
    """
    if isinstance(factor_name, string_types):
        factor_name_ = [factor_name]
    elif isinstance(factor_name, (list, tuple)):
        factor_name_ = list(factor_name)
    factor_names = factor_name_ + ['M004023']
    factor_names = [str(n) for n in factor_names]
    raw_fac = get_raw_factor(factor_names, index_code, start_date, end_date, freq)
    raw_fac = raw_fac.rename(columns={'M004023': 'cap'})
    if 'M004023' in factor_name_:
        raw_fac.loc[:, 'M004023'] = raw_fac.cap

    dts = sorted(raw_fac.index.get_level_values(0).unique())
    s, e = str(dts[0]), str(dts[-1])

    benchmark_returns = csf.get_index_hist_bar(index_code=benchmark_code, start_date=start_date, end_date=end_date,
                                               field=['close']).rename(columns={'close': 'benchmark_returns'})
    benchmark_returns.index = benchmark_returns.index.map(lambda dt: str(dt.date()))
    benchmark_returns.index.name = 'date'
    benchmark_returns = benchmark_returns.loc[dts, :]
    benchmark_returns = benchmark_returns.pct_change().shift(-1).dropna()

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

    fac_ret = fac_ret.join(benchmark_returns)

    return fac_ret


def add_group(fac_ret, num_group=5, ascending=True, method='first', na_option='keep'):
    """
    添加一个分组列
    Args:
        fac_ret (DataFrame): 一个Multi-index数据框, 含有因子,市值, 下期收益率数据, 仅支持一个因子
        num_group (int): 组数
        ascending (bool): 是否升序排列
        method (str) : {'average', 'min', 'max', 'first', 'dense'}
                            * average: average rank of group
                            * min: lowest rank in group
                            * max: highest rank in group
                            * first: ranks assigned in order they appear in the array
                            * dense: like 'min', but rank always increases by 1 between groups
        na_option (str): {'keep', 'top', 'bottom'}
                            * keep: leave NA values where they are
                            * top: smallest rank if ascending
                            * bottom: smallest rank if descending
    Returns:
        DataFrame, 比fac_ret 多了一列, 列名是group
    """

    def __add_group(frame):
        factor_name = get_factor_name(frame)

        rnk = frame[factor_name].rank(ascending=ascending, na_option=na_option, method=method)
        # 假设有k个NA, 未执行下句时, rank 值 从1..(N-k), 执行后, rnk值是从k+1..N
        rnk += rnk.isnull().sum()
        # fillna后, NA的rank被置为0.
        rnk = rnk.fillna(0.0)

        labels = ['Q{:0>2}'.format(i) for i in range(1, num_group + 1)]
        category = pd.cut(rnk, bins=num_group, labels=labels).astype(str)
        category.name = 'group'
        new_frame = frame.join(category)
        return new_frame

    if 'group' in fac_ret.columns:
        return fac_ret
    else:
        return fac_ret.groupby(level=0).apply(__add_group)


def de_extreme(fac_ret_data, num=1, method='mad'):
    return fac_ret_data.groupby(level=0).apply(extreme_process, num=num, method=method)


def standardize(fac_ret_data, method='normal'):
    return fac_ret_data.groupby(level=0).apply(data_scale, method=method)


def filter_out_st(fac_ret):
    """
    过滤出ST股票
    Args:
        fac_ret (DataFrame): 一个multi-index 数据框, level0=date, level1=code.
        基本思想是和ST股票聚合, status==null说明不是ST的
    Returns:
        DataFrame, 不包含停牌股票的fac_ret
    """
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    st_stocks = Parallel(n_jobs=20, backend='threading', verbose=5)(delayed(csf.get_st_stock_today)(dt)
                                                                    for dt in dts)
    st_stocks = pd.concat(st_stocks, ignore_index=True)
    st_stocks.loc[:, 'code'] = st_stocks.code.str.slice(0, 6)
    st_stocks = st_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(st_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result


def filter_out_suspend(fac_ret):
    dts = sorted(fac_ret.index.get_level_values(0).unique())
    suspend_stocks = Parallel(n_jobs=20, backend='threading', verbose=5)(delayed(csf.get_stock_sus_today)(date=dt)
                                                                         for dt in dts)
    for (dt, frame) in zip(dts, suspend_stocks):
        frame.loc[:, 'date'] = dt

    suspend_stocks = pd.concat(suspend_stocks, ignore_index=True)
    suspend_stocks = suspend_stocks.query('status == "T"')
    suspend_stocks = suspend_stocks.set_index(['date', 'code']).sort_index()
    joined = fac_ret.join(suspend_stocks, how='left')
    result = joined[joined.status.isnull()][fac_ret.columns]
    return result


def filter_out_recently_ipo(fac_ret, days=20):
    stocks = sorted(fac_ret.index.get_level_values(1).unique())
    ipo_info = Parallel(n_jobs=20, backend='threading', verbose=5)(
        delayed(csf.get_stock_ipo_info)(stock, field=['code', 'dt'])
        for stock in stocks)
    ipo_info = pd.concat(ipo_info, ignore_index=True)
    ipo_info.loc[:, 'code'] = ipo_info.code.str.slice(0, 6)
    ipo_info = ipo_info.rename(columns={'dt': 'listing_date'})
    fac_ret_ = fac_ret.reset_index()
    merged = pd.merge(fac_ret_, ipo_info, on='code')

    merged.loc[:, 'days'] = (merged.date.map(pd.Timestamp) - merged.listing_date.map(pd.Timestamp)).dt.days

    result = (merged.query('days>{}'.format(days))
              .set_index(['date', 'code'])
              .sort_index()[fac_ret.columns])

    return result


def raw_data_plotting():
    pass


def return_analysis(fac_ret_data):
    """
    收益率分析
    Args:
        fac_ret_data (DataFrame): 含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'
        bench_returns (Series): benchmark的收益率
    Returns:
        ReturnAnalysis
    Raises:
        ValueError, 当bench_returns index 不能包含(覆盖)fac_ret_returns
    """
    #
    # fac_index_start = fac_ret_data.index.get_level_values(0)[0]
    # fac_index_end = fac_ret_data.index.get_level_values(0)[-1]

    benchmark_returns = fac_ret_data.groupby(level=0)['benchmark_returns'].head(1).reset_index(level=1, drop=True)

    # bench_index_start = bench_returns.index[0]
    # bench_index_end = bench_returns.index[-1]
    #
    # if bench_index_start > fac_index_start or bench_index_end < fac_index_end:
    #     raise ValueError('bench_return index should contains fac_ret_data index')

    group_mean = fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data['group']])[['ret']].mean()
    group_mean = group_mean.to_panel()['ret']
    group_mean['Q_LS'] = group_mean.ix[:, 0] - group_mean.ix[:, -1]
    return_stats = pd.DataFrame()
    for col in group_mean.columns:
        return_stats[col] = return_perf_metrics(group_mean[col], benchmark_returns)

    ret = ReturnAnalysis()
    ret.benchmark_return = benchmark_returns
    ret.return_stats = return_stats
    ret.group_mean_return = group_mean
    ret.group_cum_return = (group_mean + 1).cumprod() - 1
    return ret


def information_coefficient_analysis(fac_ret_data, ic_method='normal'):
    """
    信息系数（IC）分析

    Args:
        fac_ret_data (DataFrame): 含有因子,市值,收益率,分组的数据框.且分组的列名称为'group'
        ic_method (str): ic计算方法, 有normal, rank, rank_adj
    Returns:
        ICAnalysis
    """
    factor_name = get_factor_name(fac_ret_data)
    ic_series = fac_ret_data.groupby(level=0).apply(
        lambda frame: information_coefficient(frame[factor_name], frame['ret'], ic_method))
    ic = ic_series.map(lambda e: e[0])
    p_value = ic_series.map(lambda e: e[1])
    ic_series = pd.DataFrame({'ic': ic, 'p_value': p_value})
    ic_decay = IC_decay(fac_ret_data)

    group_ic = fac_ret_data.groupby(level=0).apply(lambda frame: frame.groupby('group').apply(
        lambda df: information_coefficient(df[factor_name], df['ret'], ic_method)))
    group_ic_ic = group_ic.applymap(lambda e: e[0])
    group_ic_p_value = group_ic.applymap(lambda e: e[1])
    group_ic = pd.Panel({'ic': group_ic_ic, 'p_value': group_ic_p_value})

    ic_statistics = pd.Series({'IC_mean': ic.mean(), 'p_mean': p_value.mean(),
                               'IC_Stdev': ic.std(),
                               'IC_IR': ic.mean() / ic.std()})

    ret = ICAnalysis()
    ret.IC_series = ic_series
    ret.IC_decay = ic_decay
    ret.IC_statistics = ic_statistics
    ret.groupIC = group_ic
    return ret


def IC_decay(fac_ret_cap):
    """
    信息系数衰减, 不分组
    Args:
        fac_ret_cap (DataFrame): 一个Multiindex数据框
    :return:
    """
    grouped = fac_ret_cap.groupby(level=0)
    n = len(grouped)
    lag = min(n, 12)

    factor_name = get_factor_name(fac_ret_cap)

    rets = []
    dts = [dt for dt, _ in grouped]
    frames = (frame.reset_index(level=0, drop=True) for _, frame in grouped)
    for piece_data in window(frames, lag, longest=True):
        ret = [information_coefficient(df_fac.loc[:, factor_name], df_ret.loc[:, 'ret'])[0]
               if df_ret is not None else np.nan
               for df_fac, df_ret in zip([piece_data[0]] * lag, piece_data)]
        rets.append(ret)

    columns = [''.join(['lag', str(i)]) for i in range(lag)]
    df = pd.DataFrame(rets, index=dts[:len(rets)], columns=columns)
    decay = df.mean().to_frame()
    decay.columns = ['decay']
    return decay


def turnover_analysis(fac_ret_data, turnover_method='count'):
    """
    换手率分析
    Args:
        fac_ret_data (DataFrame): 一个Multi index 数据框, 含有factor, ret, cap, group列
        turnover_method (str): count or cap_weighted
    Returns:
        TurnoverAnalysis
    """
    ret = TurnOverAnalysis()

    # code_and_cap: index:dts, columns:groups, elements:dict, keys-->tick, values-->cap
    code_and_cap = (fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data.group])
                    .apply(lambda frame: dict(zip(frame.index.get_level_values(1), frame['cap'])))
                    .unstack()
                    )

    def __count_turnover(current_dict, next_dict):
        current_codes = set(current_dict.keys())
        next_codes = set(next_dict.keys())
        try:
            ret = len(next_codes - current_codes) * 1.0 / len(current_codes)
        except ZeroDivisionError:
            ret = np.inf
        return ret

    def __capwt_turnover(current_dict, next_dict):
        current_df = pd.Series(current_dict, name='cap').to_frame()
        current_weights = current_df.cap / current_df.cap.sum()
        next_df = pd.Series(next_dict, name='cap').to_frame()
        next_weights = next_df.cap / next_df.cap.sum()

        cur, nxt = current_weights.align(next_weights, join='outer', fill_value=0)
        ret = (cur - nxt).abs().sum() / 2
        return ret

    def auto_correlation(fac_ret_data_):

        factor_name = get_factor_name(fac_ret_data_)

        grouped = fac_ret_data_.groupby(level=0)
        n = len(grouped)
        lag = min(n, 12)
        dts = sorted(fac_ret_data.index.get_level_values(0).unique())
        group_names = sorted(grouped.groups.keys())
        table = []
        for idx in range(0, n - lag):
            rows = []
            for l in range(idx + 1, idx + 1 + lag):
                current_frame = (grouped.get_group(group_names[idx])
                                 .reset_index()
                                 .set_index('code')[factor_name].dropna())
                next_frame = (grouped.get_group(group_names[l])
                              .reset_index()
                              .set_index('code')[factor_name].dropna())
                x, y = current_frame.align(next_frame, join='inner')
                rows.append(pearsonr(x.values, y.values)[0])
            table.append(rows)
        auto_corr_ = pd.DataFrame(table, index=dts[:(n - lag)], columns=list(range(1, lag + 1)))
        return auto_corr_

    method = __count_turnover if turnover_method == 'count' else __capwt_turnover

    dts = fac_ret_data.index.get_level_values(0).unique()[:-1]
    results = {}
    for group in code_and_cap.columns:
        group_ret = []
        for idx, dic in enumerate(code_and_cap.ix[:-1, group]):
            current_dic = dic
            next_dic = code_and_cap.ix[idx + 1, group]
            group_ret.append(method(current_dic, next_dic))
        results[group] = group_ret

    turnov = pd.DataFrame(results, index=dts)

    auto_corr = auto_correlation(fac_ret_data)
    ret.auto_correlation = auto_corr
    ret.turnover = turnov
    return ret


def code_analysis(fac_ret_data, plot=False):
    """
    选股结果分析
    fac_ret_data:
    plot:

    Args:
        fac_ret_data (DataFrame):  一个Multi index 数据框, 含有factor, ret, cap, group列
    """
    ret = CodeAnalysis()

    grouped = fac_ret_data.groupby([fac_ret_data.index.get_level_values(0), fac_ret_data.group])

    # index:dt, columns:group
    stocks_per_dt_group = grouped.apply(lambda frame_: tuple(frame_.index.get_level_values(1))).unstack()

    mean_cap_per_dt_group = grouped.apply(lambda frame_: frame_['cap'].mean()).unstack()  # index:dt, columns:group

    mean_cap_per_group = mean_cap_per_dt_group.mean()

    stocks = sorted(fac_ret_data.index.get_level_values(1).unique())

    industries = [csf.get_stock_csf_industry(codes, field=['code', 'level2_name']) for codes in batch(stocks, n=90)]

    industries = pd.concat(industries)
    industries.loc[:, 'code'] = industries.code.str.slice(0, 6)
    industries_dict = dict(zip(industries.code, industries.level2_name))

    # code ---> industry
    industries_per_dt_group = stocks_per_dt_group.applymap(lambda tup: tuple(industries_dict[t] for t in tup))

    # industry tuple ---> Counter
    counter = industries_per_dt_group.applymap(lambda tup: Counter(tup))

    # counter ----> percent
    counter_percent = counter.applymap(lambda dic: {k: v * 1.0 / sum(dic.values()) for k, v in dic.iteritems()})

    dic_frame = {}
    for col in counter_percent.columns:
        frame = pd.DataFrame(counter_percent[col].tolist(), index=counter_percent.index).fillna(0)
        frame = frame[list(frame.iloc[0, :].sort_values(ascending=False).index)]
        dic_frame[col] = frame

    # 行业平均占比: 所有分组, 所有dt合并到一起
    industries_total = Counter(industries_per_dt_group.sum().sum())
    industries_total = {str(k): v for k, v in industries_total.iteritems()}
    industries_total = pd.Series(industries_total).sort_values(ascending=False)

    ret.cap_analysis = mean_cap_per_dt_group
    ret.industry_analysis = IndustryAnalysis(gp_mean_per=industries_total, gp_industry_percent=dic_frame)
    ret.stock_list = stocks_per_dt_group

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


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
