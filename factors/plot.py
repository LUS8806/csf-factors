#!/usr/bin/env python
# -*- coding: utf8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns


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
    with plt.rc_context({'figure.figsize': (18, 9)}):
        gs = gridspec.GridSpec(3, 2)
        ax1 = plt.subplot(gs[0, :])
        ax2 = plt.subplot(gs[2, 0])
        ax3 = plt.subplot(gs[2, 1])

        plot_ic_timeseries(ic_series, window=12, ax=ax1)
        plot_ic_dist(ic_series, ax=ax2)
        plot_ic_decay(ic_decay, ax=ax3)


def plot_ic_timeseries(ic_series, window=12, ax=None):
    """
    :param ic_series:
    :param window: 计算ic均值的滚动窗口
    :param ax:
    :return:
    """
    if not ax:
        ax = plt.gca()
    df = pd.DataFrame()
    df['ic_series'] = ic_series
    df['ic_mean'] = pd.rolling_mean(df['ic'], window=window)

    ax = df['ic_series'].plot(kind='bar', ax=ax, label='ic_series')
    ax = df['ic_mean'].plot(
        ax=ax, color='k', label='ic_mean(window={})'.format(window))
    xticklabels = ic_series.index.values.copy()
    N = len(ic_series)
    ax.set_xticklabels([''] * N)
    if N > 10:
        step = int(N / 10)
        xticklabels[np.arange(N) % step != 0] = ''
    ax.set_xticklabels(xticklabels)
    ax.set_title('IC TimeSeires')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    return ax


def plot_ic_decay(ic_decay, ax=None):
    """

    :param ic_decay:
    :param ax:
    :return:
    """
    if ax is None:
        ax = plt.gca()
    ic_decay.plot(kind='bar', ax=ax, title='IC Decay')
    return ax


def plot_ic_dist(ic, ax=None):
    """

    :param ic:
    :param ax:
    :return:
    """
    from scipy.stats import norm
    ax = sns.distplot(ic, fit=norm, kde=False, ax=ax)
    ax.set_title('IC Distribution')
    return ax


def plot_ret(Return_analysis):
    """
    收益率分析作图
    :param Return_analysis:  TODO 加入Benchmark Return
    :return:
    """
    cum_ret = Return_analysis.group_cum_return
    mean_ret = Return_analysis.group_mean_return

    import matplotlib.gridspec as gridspec
    with plt.rc_context({'figure.figsize': (15, 10)}):
        gs = gridspec.GridSpec(3, 3)
        ax1 = plt.subplot(gs[0, :])
        colors = plt.rcParams['axes.color_cycle'][:6]
        axes = [plt.subplot(gs[i,j]) for i in range(1,3) for j in range(3)]
        cols = [mean_ret[col] for col in mean_ret.columns]

        plot_cum_return(cum_ret, mean_ret, ax=ax1)
        for (ax, data, color, col_name) in zip(axes, cols, colors, mean_ret.columns):
            ax.hist(data, color=color)
            title = '%s mean ret dist plot'  % col_name
            ax.set_title(title)
        plt.tight_layout()


def plot_cum_return(cum_ret, mean_ret, ax=None):
    """
    累计收益率及Q_LS每期收益率作图
    :param cum_ret:
    :param mean_ret:
    :param ax:
    :return:
    """
    if not ax:
        ax = plt.gca()
#         cum_ret['benchmark'] = ben_ret.cumprod()  #TODO
    ax = cum_ret.plot(ax=ax)
    ax = mean_ret.ix[:, 'Q_LS'].plot(
        kind='bar', ax=ax, secondary_y=True, legend=True)
    xticklabels = cum_ret.index.values.copy()
    N = len(cum_ret)
    ax.set_xticklabels([''] * N)
    if N > 10:
        step = int(N / 10)
        xticklabels[np.arange(N) % step != 0] = ''
    ax.set_xticklabels(xticklabels)
    ax.set_title('Cum Return Plot')
    return ax


def plot_turnover(Turnover_analysis):
    """
    换手率作图
    :param Turnover_analysis:
    :return:
    """
    auto_cor = Turnover_analysis.auto_correlation.mean()
    buy_signal = Turnover_analysis.buy_signal
    turnover = Turnover_analysis.turnover

    plt.style.use('ggplot')
    orignal_figsize = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = (14, 4)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    turnover.plot(ax=ax1, title='turnover')

    fig_signal = plt.figure()
    ax_decay = fig_signal.add_subplot(121)
    buy_signal.decay.plot(ax=ax_decay, title='singal decay')
    ax_reversal = fig_signal.add_subplot(122)
    buy_signal.reversal.plot(ax=ax_reversal, title='signal reversal')
    plt.rcParams['figure.figsize'] = orignal_figsize


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
    industry_q = Code_analysis.industry_analysis.gp_industry_percent.loc[
        :, Q].copy()
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


def plot_insdustry_percent(df, ax=None):
    if ax is None:
        ax = plt.gca()
    # 取得各个style下面的color
    colors_ = _get_colors()
    ax = df.plot(
        kind='bar', stacked=True, color=colors_, ax=ax, width=1, alpha=0.6)
    ax, font = _show_chinese_character(ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop=font)
    ax.set_ylim([0, 1.0])
    return ax


def plot_industry_mean_percent(df, ax=None):
    colors_ = _get_colors()
    ax = df.plot(kind='bar', ax=ax, color=colors_)
    ax, _ = _show_chinese_character(ax)
    return ax


def _get_colors():
    colors_ = []
    for sty in plt.style.available:
        plt.style.use(sty)
        sty_colors = [item['color']
                      for item in list(plt.rcParams['axes.prop_cycle'])]
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