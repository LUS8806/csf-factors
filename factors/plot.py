#!/usr/bin/env python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats




def plot_ic(IC_analysis):
    """
    IC分析作图
    :param IC_analysis:
    :return:
    """
    # ic, ic_decay, auto_correlation = None, None, None
    ic_series = IC_analysis.IC_series.ic
    ic_decay = IC_analysis.IC_decay
    import matplotlib.gridspec as gridspec
    with plt.rc_context({'figure.figsize': (15, 9)}):
        gs = gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[1, 0])
        ax3 = plt.subplot(gs[1, 1])

        plot_IC_bar_and_series(ic_series, window_size=5, ax=ax1)
        plot_IC_distribution(ic_series, ax=ax2, bins=10)
        plot_IC_decay(ic_decay, ax=ax3)


def plot_ret(Return_analysis):
    """
    收益率分析作图
    :param Return_analysis:  TODO 加入Benchmark Return
    :return:
    """
    ret = Return_analysis.group_cum_return.T.copy()
    ret.plot()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(121)
    returns = ret.Q1
    plot_returns_distribution(returns, ax2, bins=10)
    ax3 = fig2.add_subplot(122)
    ret.plot(ax=ax3, rot=90)


def plot_code_result(Code_analysis):
    """
    选股结果分析作图
    :param Code_analysis:
    :return:
    """
    origin_figsize = plt.rcParams['figure.figsize']
    # cap
    plt.rcParams['figure.figsize'] = (12, 4)
    plt.style.use('ggplot')
    df = Code_analysis.cap_analysis.copy()
    industry_q = Code_analysis.industry_analysis.gp_industry_percent.loc[:, Q].copy()
    gp_industry_percent = Code_analysis.industry_analysis.gp_mean_per.copy()

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


def _plot_IC_bar(IC, ax=None):
    if ax is None:
        ax = plt.gca()
    IC.plot(kind='bar', ax=ax)
    return ax


def _plot_IC_series(IC, window_size=20, ax=None):
    if ax is None:
        ax = plt.gca()
    xticklabels = ax.get_xticklabels()
    rolling_mean = pd.rolling_mean(IC, window_size)
    rolling_mean.plot(ax=ax)
    return ax


def plot_IC_bar_and_series(IC,window_size=20,ax=None):
    if ax is None:
        ax = plt.gca()
    ax = _plot_IC_series(IC, window_size, ax)
    ax = _plot_IC_bar(IC,ax)
    xticklabels = IC.index.values.copy()
    N = len(IC)
    ax.set_xticklabels([''] * N)
    if N > 10:
        step = int(N / 10)
        xticklabels[np.arange(N)%step!=0] = ''
    ax.set_xticklabels(xticklabels)
    ax.set_title('IC')
    return ax



def plot_IC_decay(ic_decay, ax=None):
    if ax is None:
        ax = plt.gca()
    ic_decay.plot(kind='bar', ax=ax, title='IC Decay')
    return ax


def plot_IC_distribution(ic, ax=None, bins=10):
    return plot_distribution(ic, ax, bins=bins)



def plot_returns(df):
    """

    parameters: ret_dict, shoud has following keys:
    benchmark, Q1,Q2,...,Q5,value of each key is a TimeSeires

    """
    df = pd.DataFrame(ret_dict)
    df.plot()


def plot_returns_distribution(returns, ax=None, bins=10):
    return plot_distribution(returns, ax, bins=bins)


def plot_insdustry_percent(df, ax=None):
    if ax is None:
        ax = plt.gca()
    # 取得各个style下面的color
    colors_ = _get_colors()
    ax = df.plot(kind='bar',stacked=True, color = colors_,ax=ax, width=1,alpha=0.6)
    ax, font = _show_chinese_character(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=font)
    ax.set_ylim([0,1.0])
    return ax

def plot_industry_mean_percent(df,ax=None):
    colors_ = _get_colors()
    ax = df.plot(kind='bar',ax=ax, color = colors_)
    ax, _ = _show_chinese_character(ax)
    return ax


def _get_colors():
    colors_ = []
    for sty in plt.style.available:
        plt.style.use(sty)
        sty_colors = [ item['color'] for item in  list(plt.rcParams['axes.prop_cycle'])]
        colors_.extend(sty_colors)
    colors_ = list(set(colors_))
    colors_ = [c for c in colors_ if len(c) > 4]
    return colors_


def _show_chinese_character(ax):
    import os
    from matplotlib.font_manager import FontProperties
    if os.name == 'posix':
        fname = r'/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    elif os.name == 'nt':
        fname = r"c:\windows\fonts\simsun.ttc"
    font = FontProperties(fname=fname, size=10)
    
    labels = ax.get_xticklabels()+ax.legend().texts+[ax.title]

    for label in labels:
        label.set_fontproperties(font)
    return ax, font


def plot_distribution(series, ax=None, bins=10):
    if ax is None:
        ax = plt.gca()
    series.plot.hist(ax=ax, normed=1, bins=bins, alpha=0.6)
    # TODO: change bindwidth
    mean, std = series.mean(), series.std()
    min_, max_ = series.min(), series.max()
    x = np.linspace(min_, max_, len(series))
    step = (max_ - min_) / bins
    y = stats.norm.pdf(x, mean, std)
    point_x = np.linspace(min_ + step / 2, max_ - step / 2, bins)
    point_y = stats.norm.pdf(point_x, mean, std)
    ax.plot(x, y)
    ax.set_xlim([min_, max_])
    ax.set_xlabel(series.name)
    # print('type of point_x is{}'.format(type(point_x)))
    ax.scatter(point_x, point_y)
    title = ' '.join((series.name, 'distribution'))
    ax.set_title(title)
    # ax.set_xticklabels(series.index.values.tolist())
    return ax