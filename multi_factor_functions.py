#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd


def sort_and_na(df):
    df = df.rank(method='first', na_option='keep')
    df2 = df+df.isnull().sum()
    df3 = df2.fillna(0.0)
    return df3


def __equal_weighted_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_window=12,
                             fac_ret_gp="Q_LS", biggest_best=None):
    dts = standardized_factor.index.levels[0]
    columns = standardized_factor.columns
    weights = pd.DataFrame(1, index=dts, columns=columns)

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    # rank, the smallest get 1, i.e. ascending = True
    standardized_factor_copy=standardized_factor_copy.sort_index()     #先按照股票代码排序
    __sort=lambda x : sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def __frame_to_rank(biggest_best, standardized_factor):
    standardized_factor_copy = standardized_factor.copy()
    # transfer True to -1, False to 1
    asc = {}
    for k, v in biggest_best.iteritems():
        new_value = -1 if v else 1
        asc[k] = new_value
    standardized_factor_copy *= pd.Series(asc)
    return standardized_factor_copy.sort_index(ascending=False)


def __IC_weighted_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_window=12, fac_ret_gp="Q_LS",
                          biggest_best=None):
    # dts = standardized_factor.index.levels[0]
    # columns = standardized_factor.columns
    if score_window is None:
        score_window=12
    ic = {factor_name: single_factor_analysis_results[
        factor_name].IC_analysis.IC_series.ic for factor_name in factor_names}
    weights = pd.DataFrame(ic).abs()
    weights = pd.rolling_mean(weights,score_window, min_periods=1)
    # row/sum(row) for row in rows
    weights = (weights.T / weights.T.sum()).T

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    standardized_factor_copy = standardized_factor_copy.sort_index()  # 先按照股票代码排序
    __sort=lambda x : sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def __ICIR_weighted_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_window, fac_ret_gp="Q_LS",
                            biggest_best=None):
    # dts = standardized_factor.index.levels[0]
    # columns = standardized_factor.columns
    if score_window is None:
        score_window=12
    ic = {factor_name: single_factor_analysis_results[
        factor_name].IC_analysis.IC_series.ic
          for factor_name in factor_names}
    ic = pd.DataFrame(ic).abs()
    ic_roll_mean = pd.rolling_mean(ic, score_window, min_periods=1)
    ic_roll_std = pd.rolling_std(ic, score_window, min_periods=1)
    weights = ic_roll_mean / ic_roll_std
    weights.ix[0, :] = ic.ix[0, :]

    if np.inf in weights.values:
        inf_rows = weights[np.isinf(weights)].stack().unstack().index
        weights.loc[inf_rows, :] = ic_roll_mean.loc[inf_rows, :]

    # row/sum(row) for row in rows
    weights = (weights.T / weights.T.sum()).T

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    standardized_factor_copy = standardized_factor_copy.sort_index()  # 先按照股票代码排序

    __sort = lambda x: sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def __ret_weighted_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_window=12,
                           fac_ret_gp="Q_LS", biggest_best=None):
    # dts = standardized_factor.index.levels[0]
    # columns = standardized_factor.columns
    if score_window is None:
        score_window=12
    q_ls = {factor_name:
                single_factor_analysis_results[
                    factor_name].return_analysis.group_return_mean.T.loc[:,
                fac_ret_gp] for factor_name in factor_names}

    q_ls = pd.DataFrame(q_ls).abs()
    ic_roll_mean = pd.rolling_mean(q_ls, score_window, min_periods=1)
    weights = ic_roll_mean
    # row/sum(row) for row in rows
    weights = (weights.T / weights.T.sum()).T

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    standardized_factor_copy = standardized_factor_copy.sort_index()  # 先按照股票代码排序
    __sort=lambda x : sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def __retIR_weighted_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_window=12,
                             fac_ret_gp="Q_LS", biggest_best=None):
    # dts = standardized_factor.index.levels[0]
    # columns = standardized_factor.columns
    if score_window is None:
        score_window=12
    q_ls = {factor_name:
                single_factor_analysis_results[
                    factor_name].return_analysis.group_return_mean.T.loc[:,
                fac_ret_gp] for factor_name in factor_names}

    q_ls = pd.DataFrame(q_ls).abs()
    q_ls_roll_mean = pd.rolling_mean(q_ls, score_window, min_periods=1)
    q_ls_roll_std = pd.rolling_std(q_ls, score_window, min_periods=1)
    weights = q_ls_roll_mean / q_ls_roll_std
    weights.ix[0, :] = q_ls.ix[0, :]

    if np.inf in weights.values:
        inf_rows = weights[np.isinf(weights)].stack().unstack().index
        weights.loc[inf_rows, :] = q_ls_roll_mean.loc[inf_rows, :]

    # row/sum(row) for row in rows
    weights = (weights.T / weights.T.sum()).T

    standardized_factor_copy = __frame_to_rank(biggest_best, standardized_factor)
    standardized_factor_copy = standardized_factor_copy.sort_index()  # 先按照股票代码排序
    __sort=lambda x : sort_and_na(x)  # Na 的处理办法
    ret1=standardized_factor_copy.groupby(level=0).apply(__sort)
    ret = (ret1* weights).mean(axis=1)
    return ret


def factor_scoring(factor_names, standardized_factor, single_factor_analysis_results, score_method='eqwt',
                   score_window=12, fac_ret_gp="Q_LS", biggest_best=None):
    """
    :param biggest_best, a dict, keys are factor names, value is bool.
    """
    # for a factor, if the biggest is the best, default the smallest one got rank = 1.
    valid_methods = {'eqwt': __equal_weighted_scoring,
                     'ICwt': __IC_weighted_scoring,
                     'ICIRwt': __ICIR_weighted_scoring,
                     'retwt': __ret_weighted_scoring,
                     'retIRwt': __retIR_weighted_scoring}

    ret = valid_methods[score_method](factor_names, standardized_factor,
                                      single_factor_analysis_results,
                                      score_window=score_window,
                                      fac_ret_gp=fac_ret_gp,
                                      biggest_best = biggest_best
                                      )
    ret.name = 'factors_score'
    return ret.sort_index()
