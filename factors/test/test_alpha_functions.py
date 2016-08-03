#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
from collections import OrderedDict
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal

from alpha.alpha_algo.alpha_functions import (get_grouped_data)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
from numpy.testing.utils import assert_allclose


def generate_group_data(fac_ret_cap, factor_name, num_group):
    cols = [factor_name, 'ret', 'cap']
    for dt, frame in fac_ret_cap.iteritems():
        for group_name, group_data in origin_cut_group(frame.loc[:, cols],
                                                       num_group=num_group):
            yield dt, group_name, group_data


def origin_cut_group(df, include_index=True, num_group=5, is_ascend=0):
    """
    将数据分组
    Args:
        df: 对每个时间戳对应的数据框进行分组， index为股票代码，columns为fac1,fac2,...,ret,cap
        include_index:
        num_group:
        is_ascend:

    Returns:
        grouped data
    """
    # TODO: 处理include_index
    bins = np.linspace(0, len(df), num_group + 1)
    digits_group = np.digitize(range(len(df)), bins)
    groups = [''.join(['Q', str(i)]) for i in digits_group]
    df.index.name = 'codes'
    factor_name = list(set(df.columns) - {'ret', 'codes', 'cap'})
    assert len(
        factor_name) == 1, "len(factor_name) should be 1, but factor_name is {}".format(
        factor_name)
    df = df.reset_index()
    df.sort_values(by=[factor_name[0], 'codes'], ascending=[is_ascend, True],
                   inplace=True)
    for group_name, data in df.groupby(groups):
        yield group_name, data


class TestAlphaFunctions(TestCase):
    @classmethod
    def setUpClass(cls):
        from ..csf_alpha import CSFAlpha
        import glob
        fac_lst = ['base.roae', 'p.p1']
        factor_name = fac_lst[0]
        start_date = '2013-01-01'  # 开始日期
        end_date = '2015-12-31'  # 结束日期
        bench_code = '000905'  # benchmark代码
        ins = CSFAlpha(bench_code, None, fac_lst, start_date, end_date,
                       bench_code=bench_code, freq='m', isIndex=True)
        ins.get_all_data()
        ins.parallel_run_single_factor_analysis(n_jobs=4)
        cls.results = ins.single_factor_analysis_results[factor_name]
        cls.all_data = ins.all_data
        cls.all_stocks = ins.all_stocks
        cls.fac_lst = fac_lst
        cls.freq = ins.freq
        cls.return_mean_method = ins.return_mean_method
        cls.num_group = ins.num_group
        cls.ic_method = ins.ic_method
        cls.ascending = False
        cls.fp_month = ins.fp_month
        cls.g_buy = ins.g_buy
        cls.g_sell = ins.g_sell
        cls.turnover_method = ins.turnover_method

        FILES = glob.glob('*.csv')
        for f in FILES:
            base_name = os.path.basename(f).split('.')[0]
            setattr(cls, base_name, pd.read_csv(f, index_col=[0]))

    def test_cut_group(self):
        origin_fac_ret_cap = OrderedDict()
        for dt, frame in self.all_data['fac_ret_cap'].groupby(level=0):
            tmp = frame.reset_index(level=0, drop=True)
            origin_fac_ret_cap[dt] = tmp
        origin_groups = generate_group_data(origin_fac_ret_cap,
                                            num_group=self.num_group,
                                            factor_name=self.fac_lst[0])
        origin_groups = list(origin_groups)
        grouped = get_grouped_data(fac_ret_cap=self.all_data['fac_ret_cap'],
                                   factor_name=self.fac_lst[0],
                                   num_group=self.num_group, ascending=False)
        for idx, group in enumerate(grouped):
            np.testing.assert_equal(set(group[1].reset_index().secu),
                                    set(origin_groups[idx][2].codes))

    def test_group_return_cumulative(self):
        ret = self.results.return_analysis.group_return_cumulative
        benchmark = self.group_return_cumulative
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_group_return_mean(self):
        ret = self.results.return_analysis.group_return_mean
        benchmark = self.group_return_mean
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_return_statistics(self):
        ret_cols = self.results.return_analysis.return_statistics.columns
        ret = self.results.return_analysis.return_statistics.loc[:,
              sorted(ret_cols)]

        benchmark_cols = self.return_statistics.columns
        benchmark = self.return_statistics.loc[:, sorted(benchmark_cols)]
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_IC_series(self):
        ret = self.results.IC_analysis.IC_series
        benchmark = self.IC_series
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_groupIC_IC(self):
        ret = self.results.IC_analysis.groupIC.ix['ic']
        benchmark = self.groupIC_ic
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_groupIC_p_value(self):
        ret = self.results.IC_analysis.groupIC.ix['p_value']
        benchmark = self.groupIC_p_value
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_IC_decay(self):
        ret = self.results.IC_analysis.IC_decay
        benchmark = self.IC_decay
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_buy_signal(self):
        ret = self.results.turnover_analysis.buy_signal
        benchmark = self.buy_signal
        assert_frame_equal(ret, benchmark, check_names=False)

    def test_auto_correlation(self):
        ret = self.results.turnover_analysis.auto_correlation
        benchmark = self.auto_correlation
        assert_frame_equal(ret.loc[benchmark.index, :], benchmark,
                           check_names=False)

    def test_phil_turnover(self):
        ret = self.results.turnover_analysis.turnover
        benchmark = self.turnover
        assert_frame_equal(ret, benchmark, check_names=False)
