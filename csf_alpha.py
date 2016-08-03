# -*- coding: UTF-8 -*-
from __future__ import division

from functools import partial

from config import *
from get_data import get_benchmark_return, get_raw_factor, get_term_return, get_cap_data


import pickle
import alpha.util as ut
from plot import *
from util import (scale, handle_extreme, GetSql, form_dt_index,
                  generate_report_dates, FactorData, ReturnAnalysis,
                  ICAnalysis, TurnOverAnalysis, CodeAnalysis)
from collections import OrderedDict
from scipy.stats import chi2_contingency
from alpha_functions import (
    phil_single_factor_analysis as single_factor_analysis,
    phil_turnover_analysis as turnover_analysis,
    phil_ic_analysis as ic_analysis,
    phil_return_analysis as return_analysis,
    phil_code_analysis as code_analysis)
from multi_factor_functions import factor_scoring
import logging
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def _standardize_raw_factor(df, ex_method, st_method):
    """
    因子数据处理：极值处理--标准化
    """
    v1 = handle_extreme(df, num=3, method=ex_method)
    # print v1
    v2 = scale().scale(v1, method=st_method)
    # print v2.sort_values(by = 'p.p1', ascending=False)
    return v2


class CSFAlpha(object):
    """
    获取测试所需数据：
    1. 原始因子数据： df_raw_factor, DataFrame格式
    2. 处理后的因子数据： standardized_factor, {dt: DataFrame}
    3. 股票收益率数据： stock_term_return,
    4. 流通市值数据： stock_mkt_caps
    5. 比较标准收益率数据： bench_ret_term
    6. 综合数据（用于分析）： df_com_facs
    """

    def __init__(self, codes, factor_codes, start_date, end_date,
                 bench_code, freq='m', isIndex=True, turnover_method='count',
                 ic_method='normal', return_mean_method='eqwt', num_group=5,
                 fp_month=12, g_buy='Q1', g_sell='Q5', sam_level=1,
                 remove_extreme_value_method='mad', scale_method='normal',ascending=None):
        """
        TODO: freq 和 fp_month 互动， 只要确定一个参数即可；
        g_buy, g_sell 和num_group互动
        Args:
            factor_codes:
            codes:       股票代码列表或指数代码
            factor_codes:     因子代码列表
            start_date:  开始日期
            end_date:    结束日期
            freq:        调仓频率(m for month, w for week, q for season)
            isIndex:
            turnover_method:
            ic_method:
            return_mean_method:
            num_group:
            g_buy:
            g_sell:
            sam_level:
            remove_extreme_value_method: std, mad, etc.
            scale_method: normal, cap, sector, etc.


        Returns:

        """
        self.fp_month = fp_month
        self.isIndex = isIndex  # 输入是否为指数代码
        self.turnover_method = turnover_method
        self.ic_method = ic_method
        self.return_mean_method = return_mean_method
        self.num_group = num_group
        self.g_buy = g_buy
        self.g_sell = g_sell
        self.sam_level = sam_level
        self.factor_codes = factor_codes
        self.codes_factor_dict = self.get_codes_factor_dict(factor_codes)  # 需要测试的因子代码
        # 按默认方向排序
        fac_info = pd.read_csv(FACTORS_DETAIL_PATH)
        self.ascending = dict(fac_info[['code','ascend']][fac_info['code'].isin(self.factor_codes)].values)
        if ascending is not None:
            self.ascending.update(ascending)
        # 防止传入的ascending的key 不在factor_codes中
        # self.ascending = {k:v for k,v in self.ascending.iteritems() if k in self.factor_codes}

        self.freq = freq  # 回测的频率
        self.start_date = start_date  # 回测开始日期
        self.end_date = end_date  # 回测结束日期
        self.bench_code = bench_code  # 比较标准的指数代码
        self.get_sql = GetSql()  # sql数据获取接口
        self.dt_index = form_dt_index(start_date, end_date, freq,
                                      END_TD_PATH)  # 调仓日期
        self.codes = codes  # 回测的指数代码
        # 每个调仓期的股票代码，字典格式
        self.term_stock_codes = self._get_idx_his_stock() if isIndex else codes
        self.rpt_terms = generate_report_dates(self.dt_index)  # 涉及到的财务报告期
        # 所有调仓期涉及的股票
        self.all_stocks = self._get_all_stock() if isIndex else codes
        self.all_data = {}  # 存放所有不同结构因子数据的字典
        # 存放所有因子测试的详细结果
        self.single_factor_analysis_results = dict(
            zip(self.factor_codes, [FactorData(None,
                                               ReturnAnalysis(),
                                               ICAnalysis(),
                                               TurnOverAnalysis(),
                                               CodeAnalysis())] * len(
                self.factor_codes)))
        self.multi_factor_analysis_results = {}  # 多因子组合测试的详细结果
        self.remove_extreme_value_method = remove_extreme_value_method
        self.scale_method = scale_method

    @staticmethod
    def get_codes_factor_dict(factor_codes):
        fac_info = pd.read_csv(FACTORS_DETAIL_PATH)
        fac_info = fac_info[fac_info.stat == 1]
        ltm_mask = fac_info.tb == 'metrics.comm_idx_quant_his_a'
        ytd_mask = fac_info.tb == 'metrics.comm_idx_quant_ytd_his_a'
        fd = fac_info.loc[:, 'fd']
        fd[ltm_mask] += '_ltm'
        fd[ytd_mask] += '_ytd'
        code_pos_dict = dict(zip(fac_info.code, fd))
        ret = {code: code_pos_dict.get(code) for code in factor_codes}
        return ret


    def _get_all_stock(self):
        """
        获取所有调仓期涉及到的所有股票代码
        @返回：
        ret: list, 股票代码列表
        """

        ret = list(set(self.term_stock_codes.levels[1]))
        return ret

    def _get_idx_his_stock(self):
        """
        获取所选指数self.codes每一个调仓期的历史成份股
        调用ut模块的函数
        @返回：
        ret: dict，{dt: stock_list}
        """
        # ret = OrderedDict()
        # for dt in self.dt_index[0:-1]:
        #     ret[dt] = ut.get_index_components(self.codes, dt)
        # return ret


        rets = []
        for dt in self.dt_index[:-1]:
            ret = ut.get_index_components(self.codes, dt)
            ret = zip([dt] * len(ret), ret)
            rets.extend(ret)
        return pd.MultiIndex.from_tuples(rets, names=['dt', 'secu'])

    def _get_raw_factor(self):  #
        ret = get_raw_factor(FACTORS_DETAIL_PATH, self.factor_codes,
                             self.rpt_terms,
                             self.all_stocks, self.dt_index,
                             self.term_stock_codes)

        # rename factor name to code, i.e. p.p1 ---> MXXXX
        factor_codes_dict = dict(zip(self.codes_factor_dict.values(), self.codes_factor_dict.keys()))
        columns_to_rename = {name:factor_codes_dict.get(name) for name in ret.columns}
        ret = ret.rename(columns=columns_to_rename)

        # 有一些列全是NA， 要去掉
        na_columns_mask = (ret.isnull().mean() == 1)
        if na_columns_mask.any():
            all_na_columns = ret.columns[na_columns_mask].tolist()
            logger.info('column {} are removed due to ALL NA'.format(all_na_columns))

            ret = ret.drop(all_na_columns, axis=1)
            self.factor_codes = sorted(set(self.factor_codes) - set(all_na_columns))
            for k in all_na_columns:
                self.ascending.pop(k)

        self.all_data['df_raw_factor'] = ret
        return ret

    # @profile
    @ut.timethis
    def _get_standardized_factor(self):
        """
        获取处理后的因子数据
        @返回：
        dic_h：dict,{dt: df}
        """
        if 'df_raw_factor' not in self.all_data:
            df = self._get_raw_factor()
        else:
            df = self.all_data['df_raw_factor']

        # dic_h = OrderedDict()
        # for dt in self.dt_index[0:-1]:
        #     df = df[dt].copy()
        #     dic_h[dt] = self._standardize_raw_factor(df, ex_method=self.remove_extreme_value_method,
        #                                              st_method=self.scale_method)
        # self.all_data['standardized_factor'] = dic_h
        # return dic_h
        standardize_raw_factor = partial(_standardize_raw_factor,
                                         ex_method=self.remove_extreme_value_method,
                                         st_method=self.scale_method)
        standardize_raw_factor.__module__ = _standardize_raw_factor.__module__
        standardize_raw_factor.__name__ = _standardize_raw_factor.__name__

        df = df.groupby(level=0).apply(standardize_raw_factor)  # group by dt
        self.all_data['standardized_factor'] = df
        return df

    # # @timethis
    def _get_term_return(self):
        """
        逐个调仓期,根据当期的股票代码（历史成份股记录）读取下期收益率
        收益率 = (下期股价-当期股价)/当期股价
        @返回：
        ret: dict, {dt: df}
        """
        self.all_data['stock_term_return'] = get_term_return(self.all_stocks,
                                                             self.dt_index,
                                                             self.term_stock_codes)
        return self.all_data['stock_term_return']

    # # @timethis
    def _get_cap_data(self):
        """
        获取流通市值数据: 字段tfc,表comm_idx_price_his_a
        @返回：
        ret: dict, {dt: df}
        """
        ret = get_cap_data(self.dt_index, self.term_stock_codes, self.all_stocks)
        self.all_data['stock_mkt_cap'] = ret
        return ret

    def combine_fac_ret_cap(self):
        """
        合并因子(去极值、标准化后的)
        Returns:

        """
        if not self.all_data:
            self.get_all_data()
        facs = self.all_data['standardized_factor']
        rets = self.all_data['stock_term_return']  # standardized_factor已经有了
        caps = self.all_data['stock_mkt_cap']

        # combined = OrderedDict()
        # for dt, fac in facs.iteritems():
        #     combined[dt] = pd.concat([fac, caps[dt], rets[dt]], axis=1)

        combined = pd.concat([facs, caps, rets], axis=1)
        combined = combined.sort_index()
        self.all_data['fac_ret_cap'] = combined
        return combined

    def _get_benchmark_return(self):
        self.all_data['benchmark_term_return'] = get_benchmark_return(
            self.bench_code, self.dt_index)
        return self.all_data['benchmark_term_return']

    @ut.timethis
    def get_all_data(self):
        """
        获取所有需要的数据：
        原始因子数据：df_raw_factor
        处理后因子数据： standardized_factor
        收益率数据: stock_term_return
        流通市值: stock_mkt_cap
        基准收益率数据： benchmark_term_return
        综合数据: df_com_facs
        """
        data_name_dic = OrderedDict([
            ('df_raw_factor', self._get_raw_factor),
            ('standardized_factor', self._get_standardized_factor),
            ('stock_term_return', self._get_term_return),
            ('stock_mkt_cap', self._get_cap_data),
            ('benchmark_term_return', self._get_benchmark_return)
        ])

        for func_name, func in data_name_dic.iteritems():
            func()

        self.combine_fac_ret_cap()


    def _regress_analysis(self, df_com_fac):
        """
        基于线性回归的指标
        """
        ret = OrderedDict()
        for dt, df_v in df_com_fac.iteritems():
            reg_ans = pd.ols(y=df_v['ret'], x=df_v['f_v'])
            ret[dt] = [reg_ans.beta['x'], reg_ans.t_stat['x']]
        ret = pd.DataFrame(ret)
        return ret

    # @profile
    def _chi_2_test(self, df_com_fac):
        """
        [TODO]: Check
        开方检验，
        返回开方检验值、开方检验p-value以及cramer's value
        计算cramer's value
        """
        ret = OrderedDict()
        for dt, df_v in df_com_fac.iteritems():
            print "dt: %s" % dt
            # print df
            df_ret = df_v.ix[:, ['codes', 'ret']]
            df_f = df_v.ix[:, ['codes', 'f_v']]
            gp_ret = self._group_cut_fun(
                df_ret, key_col='ret', include_index=False)
            gp_f = self._group_cut_fun(
                df_f, key_col='f_v', include_index=False)
            a11 = len(
                [i for i in gp_f['Q1']['codes'] if i in gp_ret['Q1']['codes']])
            a12 = len(
                [i for i in gp_f['Q1']['codes'] if i in gp_ret['Q5']['codes']])
            a21 = len(
                [i for i in gp_f['Q5']['codes'] if i in gp_ret['Q1']['codes']])
            a22 = len(
                [i for i in gp_f['Q5']['codes'] if i in gp_ret['Q5']['codes']])
            arr = np.array([a11, a12, a21, a22]).reshape(2, 2)
            ret[dt] = chi2_contingency(arr)[0]
        return ret

    # +++++单因子测试显示函数++++

    # 3.0 多因子分析
    # 因子打分
    def factor_scoring(self, fac_names, score_method='eqwt', score_window=12,
                       fac_ret_gp='Q_LS', biggest_best=None):
        """
        因子打分
        Args:
        score_method, 因子权重计算方法
        'eqwt': 等权法
        'ICwt': IC加权法，根据因子过去12个月IC均值计算每期每个因子的权重
        'ICIRwt': ICIR加权法，根据因子过去12个月ICIR均值计算每期每个因子的权重
        'retwt': 因子收益率加权法，定义Q_LS为因子收益， 根据过去12个月因子收益率均值确定权重
        'retIRwt': 因子收益率IR加权法，定义Q_LS为因子收益，根据过去12个月因子收益率IR确定权重

        'score_window': eqwt之外加权法用到的滑窗大小
        'fac_ret_gp': 代表因子收益率的组别，默认为Q_LS(Q5-Q1)
        Returns:
        df_com_socre (dt,secu) multi-indexed series.
        """
        if not self.all_data:
            self.get_all_data()
        data = self.all_data['standardized_factor'].loc[:, fac_names]

        key0 = self.single_factor_analysis_results.keys()[0]
        #  判断single_factor_analysis
        if not self.single_factor_analysis_results[key0].name:
            self.parallel_run_single_factor_analysis(n_jobs=1)
        results = self.single_factor_analysis_results
        results={k: v for k, v in results.iteritems() if k in fac_names}
        ret = factor_scoring(factor_names=fac_names, standardized_factor=data, single_factor_analysis_results=results,
                             score_method=score_method, score_window=score_window, fac_ret_gp=fac_ret_gp, biggest_best=biggest_best)
        return ret

    def multi_factor_analysis(self, fac_names=None,num_group=None, comb_name=None, score_method='eqwt', score_window=None,
                              biggest_best=None):
        """
        fac_names: 多因子组合的因子代码列表
        comb_name: 自定义组合名称

        Args:
            biggest_best:
        """
        if biggest_best==None:
            biggest_best=self.ascending
        if fac_names==None:
            fac_names=self.factor_codes
        if num_group==None:
            num_group=self.num_group
        df_com_score = self.factor_scoring(
            fac_names, score_method,
            score_window, fac_ret_gp='Q_LS', biggest_best=biggest_best)

        if comb_name is not None:
            df_com_score.name = comb_name

        name = df_com_score.name

        df_com_score = pd.concat([df_com_score,
                                  self.all_data['fac_ret_cap'].loc[:,
                                  ['ret', 'cap']]], axis=1)

        ans = single_factor_analysis(factor_name=name,
                                     fac_ret_cap=df_com_score,
                                     freq=self.freq,
                                     g_sell=self.g_sell,
                                     benchmark_term_return=self.all_data[
                                         'benchmark_term_return'],
                                     num_group=num_group,
                                     all_stocks=self.all_stocks, ascending=False)
        self.multi_factor_analysis_results[name] = ans
        return ans

    def factor_category(self):
        """
        因子分类
        """
        pass

    def analysis_plot(res_fac_analysis):
        """
        TODO: 分析结果画图, 包括分布图
        """
        # ret_analysis_plot

        pass

    def _plot_ic(self, fac_name='p.p1', is_comb= False, comb_name='comb_name'):
        """
        假设此时self.single_fac_analysis_results存在
        """
        # TODO: check
        # ic, ic_decay, auto_correlation = None, None, None
        if is_comb:
            ic = self.multi_factor_analysis_results[
                comb_name].IC_analysis.IC_series.ic
            ic_decay = self.multi_factor_analysis_results[
                comb_name].IC_analysis.IC_decay
            auto_correlation = self.multi_factor_analysis_results[
                comb_name].turnover_analysis.auto_correlation
        else:
            ic = self.single_factor_analysis_results[
                fac_name].IC_analysis.IC_series.ic
            ic_decay = self.single_factor_analysis_results[
                fac_name].IC_analysis.IC_decay
            auto_correlation = self.single_factor_analysis_results[
                fac_name].turnover_analysis.auto_correlation
        import matplotlib.gridspec as gridspec
        with plt.rc_context({'figure.figsize': (15, 9)}):
            gs = gridspec.GridSpec(3, 2)
            ax1 = plt.subplot(gs[0, :])
            ax2 = plt.subplot(gs[1, 0])
            ax3 = plt.subplot(gs[1, 1])
            ax5 = plt.subplot(gs[2, 1])

            plot_IC_bar_and_series(ic, window_size=5, ax=ax1)
            plot_IC_distribution(ic, ax=ax2, bins=10)
            plot_IC_decay(ic_decay, ax=ax3)
            auto_correlation.mean().plot.bar(ax=ax5, title='auto correlation')

    def _plot_ret(self, fac_name='p.p1',is_comb= False,comb_name='comb_name'):
        if is_comb:
            ret = self.multi_factor_analysis_results[comb_name].return_analysis.group_return_cumulative.T.copy()
        else:
            ret = self.single_factor_analysis_results[
                fac_name].return_analysis.group_return_cumulative.T.copy()
        ret.loc[:, 'benchmark'] = self.all_data['benchmark_term_return']
        ret.plot()

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(121)
        # TODO: javascript
        returns = ret.Q1
        plot_returns_distribution(returns, ax2, bins=10)

        ax3 = fig2.add_subplot(122)
        ret.plot(ax=ax3, rot=90)

    def _plot_code_result(self, fac_name='p.p1', Q='Q1',is_comb= False,comb_name='comb_name'):
        origin_figsize = plt.rcParams['figure.figsize']
        # cap
        plt.rcParams['figure.figsize'] = (12, 4)
        plt.style.use('ggplot')
        if is_comb:
            df=self.multi_factor_analysis_results[comb_name].code_analysis.cap_analysis.copy()
            industry_q = self.multi_factor_analysis_results[
                             comb_name].code_analysis.industry_analysis.gp_industry_percent.loc[
                         :, Q].copy()
            gp_industry_percent = self.multi_factor_analysis_results[comb_name].code_analysis.industry_analysis.gp_mean_per.copy()
        else:
            df = self.single_factor_analysis_results[fac_name].code_analysis.cap_analysis.copy()
            industry_q = self.single_factor_analysis_results[
                 fac_name].code_analysis.industry_analysis.gp_industry_percent.loc[:, Q].copy()
            gp_industry_percent = self.single_factor_analysis_results[fac_name].code_analysis.industry_analysis.gp_mean_per.copy()
        mean_cap = df.loc[:, 'group_cap_mean']
        df = df.drop(['group_cap_mean'], axis=1)
        df = df.T.loc[:, Q]
        fig = plt.figure()
        ax_cap = fig.add_subplot(121)
        df.plot(kind='area', ax=ax_cap, title=df.name, fontsize=10)
        ax_mean_cap = fig.add_subplot(122)
        mean_cap.plot(
            kind='bar', ax=ax_mean_cap, title='Average cap', fontsize=10)
        plt.subplots_adjust(bottom=0.0, left=.01, top=0.9, right=1.5)
        # industry
        fig2 = plt.figure()
        df = pd.DataFrame(industry_q.to_dict()).fillna(0).T
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3)
        ax1 = fig2.add_subplot(gs[0, :2])
        plot_insdustry_percent(df, ax=ax1)
        ax2 = plt.subplot(gs[0, 2])
        gp_industry_percent = gp_industry_percent.fillna(0)
        df2 = pd.DataFrame(gp_industry_percent.loc[:, Q])
        plot_industry_mean_percent(df2, ax=ax2)
        plt.subplots_adjust(bottom=0, left=.01, right=1.5, wspace=0.5)
        # plt.tight_layout(w_pad=8)
        plt.rcParams['figure.figsize'] = origin_figsize

    def _plot_turnover(self, fac_name='p.p1',is_comb= False,comb_name='comb_name'):
        if is_comb:
            buy_signal = self.multi_factor_analysis_results[
                comb_name].turnover_analysis.buy_signal
            turnover = self.multi_factor_analysis_results[
                comb_name].turnover_analysis.turnover
        else:
            buy_signal = self.single_factor_analysis_results[
                fac_name].turnover_analysis.buy_signal
            turnover = self.single_factor_analysis_results[
                fac_name].turnover_analysis.turnover
        plt.style.use('ggplot')
        # turnover
        orignal_figsize = plt.rcParams['figure.figsize']
        plt.rcParams['figure.figsize'] = (14, 4)
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        turnover.plot(ax=ax1, title='turnover')

        # average turnover & turnover compare

        # signal decay & signal reverse
        fig_signal = plt.figure()
        ax_decay = fig_signal.add_subplot(121)
        buy_signal.decay.plot(ax=ax_decay, title='singal decay')
        ax_reversal = fig_signal.add_subplot(122)
        buy_signal.reversal.plot(ax=ax_reversal, title='signal reversal')
        plt.rcParams['figure.figsize'] = orignal_figsize

    @ut.timethis
    def parallel_run_single_factor_analysis(self, n_jobs=1):
        if 'fac_ret_cap' not in self.all_data:
            self.combine_fac_ret_cap()

        ret = Parallel(n_jobs=n_jobs)(
            delayed(single_factor_analysis)(
                factor_name,
                fac_ret_cap=self.all_data[
                    'fac_ret_cap'],
                freq=self.freq,
                g_sell=self.g_sell,
                benchmark_term_return=
                self.all_data[
                    'benchmark_term_return'],
                turnover_method=self.turnover_method,
                ic_method=self.ic_method,
                return_mean_method=self.return_mean_method,
                num_group=self.num_group,
                fp_month=self.fp_month,
                g_buy=self.g_buy,
                sam_level=self.sam_level,
                all_stocks=self.all_stocks,
                ascending=self.ascending[factor_name]
            ) for factor_name in self.factor_codes)

        ret = dict(zip(self.factor_codes, ret))
        self.single_factor_analysis_results = ret
        return ret

    def return_analysis(self, n_jobs=1):
        ret = Parallel(n_jobs=n_jobs)(
            delayed(return_analysis)(
                factor_name,
                fac_ret_cap=self.all_data[
                    'fac_ret_cap'],
                freq=self.freq,
                benchmark_term_return=self.all_data[
                    'benchmark_term_return'],
                num_group=self.num_group,
                ascending=self.ascending[factor_name],
                return_mean_method=self.return_mean_method)
            for factor_name in self.factor_codes)
        return_analysis_dic = dict(zip(self.factor_codes, ret))
        for factor_name in self.factor_codes:
            self.single_factor_analysis_results[
                factor_name].reutrn_analysis = return_analysis_dic[factor_name]

        return return_analysis_dic

    def IC_analysis(self, n_jobs=1):
        ret = Parallel(n_jobs=n_jobs)(
            delayed(ic_analysis)(factor_name,
                                 fac_ret_cap=self.all_data[
                                     'fac_ret_cap'],
                                 freq=self.freq,
                                 num_group=self.num_group,
                                 ascending=self.ascending[factor_name],
                                 ic_method=self.ic_method)
            for factor_name in self.factor_codes)
        ic_analysis_dic = dict(zip(self.factor_codes, ret))
        for factor_name in self.factor_codes:
            self.single_factor_analysis_results[
                factor_name].ic_analysis = ic_analysis_dic[factor_name]

        return ic_analysis_dic

    def turnover_analysis(self, n_jobs=1):
        ret = Parallel(n_jobs=n_jobs)(
            delayed(turnover_analysis)(
                factor_name,
                fac_ret_cap=self.all_data['fac_ret_cap'],
                fp_month=self.fp_month,
                g_buy=self.g_buy,
                g_sell=self.g_sell,
                num_group=self.num_group,
                ascending=self.ascending[factor_name],
                turnover_method=self.turnover_method)
            for factor_name in self.factor_codes)
        turnover_analysis_dic = dict(zip(self.factor_codes, ret))
        for factor_name in self.factor_codes:
            self.single_factor_analysis_results[
                factor_name].reutrn_analysis = turnover_analysis_dic[
                factor_name]
        return turnover_analysis_dic

    def code_analysis(self, n_jobs=1):
        ret = Parallel(n_jobs=n_jobs)(
            delayed(code_analysis)(
                factor_name,
                grouped=self.all_data['fac_ret_cap'],
                all_stocks=self.all_stocks,
                num_group=self.num_group,
                sam_level=self.sam_level,
                ascending=self.ascending[factor_name]
            ) for factor_name in self.factor_codes)
        code_analysis_dic = dict(zip(self.factor_codes, ret))
        for factor_name in self.factor_codes:
            self.single_factor_analysis_results[
                factor_name].code_analysis = code_analysis_dic[factor_name]
        return code_analysis_dic


if __name__ == '__main__':
    # factor_codes = ['base.roae', 'p.p1', 'items.macd', 'items.tsi']  # 因子列表
    fac_info = pd.read_csv(FACTORS_DETAIL_PATH)
    fac_info = fac_info[fac_info['stat'] == 1]
    factor_codes = fac_info.code.tolist()[:24]
    # factor_codes = ['M001019','M001001','M001003']
    start_date = '2007-01-01'  # 开始日期
    # start_date = '2013-01-01'  # 开始日期
    end_date = '2008-02-28'  # 结束日期
    bench_code = '000300'  # benchmark代码
    ins = CSFAlpha(bench_code, factor_codes, start_date, end_date,
                   bench_code=bench_code, freq='m', isIndex=True, num_group=5,
                   g_sell='Q5')
    # ins.get_all_data()
    import multiprocessing
    n_jobs = multiprocessing.cpu_count()
    ret = ins.parallel_run_single_factor_analysis(n_jobs=n_jobs)

    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    ins.multi_factor_analysis(fac_names=factor_codes, comb_name='comb_name')
    ins._plot_ic(comb_name='comb_name', is_comb=True)
    # ins._plot_ret( fac_name=['p.p7',"p.p1"],is_comb=True ,comb_name='comb_name')
    # ins._plot_code_result(fac_name=['p.p7',"p.p1"], Q='Q1',is_comb= True,comb_name='comb_name')
    # ins._plot_turnover(is_comb=True,comb_name='comb_name')
    plt.tight_layout()
    plt.show()

    print 'done!'
    print 'done!'
    print 'done!'
