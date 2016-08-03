#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

import pandas as pd
import numpy as np
from scipy import stats, optimize


APPROX_BDAYS_PER_MONTH = 21
APPROX_BDAYS_PER_YEAR = 252

MONTHS_PER_YEAR = 12
WEEKS_PER_YEAR = 52
QUARTERS_PER_YEAR = 4

DAILY = 'daily'
WEEKLY = 'weekly'
MONTHLY = 'monthly'
YEARLY = 'yearly'
QUARTERLY = 'quarterly'


ANNUALIZATION_FACTORS = {
    DAILY: APPROX_BDAYS_PER_YEAR,
    WEEKLY: WEEKS_PER_YEAR,
    MONTHLY: MONTHS_PER_YEAR,
    QUARTERLY: QUARTERS_PER_YEAR
}


# Return Analysis Metrics
def annualization_factor(period, annualization):
    """
    Determine the annualization factor
    
    :param period: str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization: int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Annualization factor.
    """
    if annualization is None:
        try:
            factor = ANNUALIZATION_FACTORS[period]
        except KeyError:
            raise ValueError(
                "Period cannot be '{}'. "
                "Can be '{}'.".format(
                    period, "', '".join(ANNUALIZATION_FACTORS.keys())
                )
            )
    else:
        factor = annualization
    return factor


def cum_returns(returns, starting_value=0):
    """
    Compute cumulative returns from simple returns.

    :param returns : pd.Series
        Returns of the strategy as a percentage, noncumulative.
        - Time series with decimal returns.
        - Example:
            2015-07-16    -0.012143
            2015-07-17    0.045350
            2015-07-20    0.030957
            2015-07-21    0.004902.
    :param starting_value : float, optional
        The starting returns.

    :return pandas.Series
        Series of cumulative returns.
    """
    if pd.isnull(returns.iloc[0]):
        returns.iloc[0] = 0.
    df_cum = np.exp(np.log1p(returns).cumsum())
    if starting_value == 0:
        return df_cum - 1
    else:
        return df_cum * starting_value


def aggregate_returns(returns, convert_to):
    """
    Aggregates returns by week, month, or year.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param convert_to : str
        Can be 'weekly', 'monthly', or 'yearly'.

    :return pd.Series
        Aggregated returns.
    """
    def cumulate_returns(x):
        return cum_returns(x)[-1]
    if convert_to == WEEKLY:
        grouping = [lambda x: x.year, lambda x: x.isocalendar()[1]]
    elif convert_to == MONTHLY:
        grouping = [lambda x: x.year, lambda x: x.month]
    elif convert_to == YEARLY:
        grouping = [lambda x: x.year]
    else:
        raise ValueError(
            'convert_to must be {}, {} or {}'.format(WEEKLY, MONTHLY, YEARLY)
        )
    return returns.groupby(grouping).apply(cumulate_returns)


def max_drawdown(returns):
    """
    Determines the maximum drawdown of a strategy.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.

    :return float
        Maximum drawdown.
    """
    if len(returns) < 1:
        return np.nan

    cumulative = cum_returns(returns, starting_value=100)
    max_return = cumulative.cummax()
    return cumulative.sub(max_return).div(max_return).min()


def annual_return(returns, period=DAILY, annualization=None):
    """
    Determines the mean annual growth rate of returns.

    :param returns : pd.Series
        Periodic returns of the strategy, noncumulative.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Annual Return as CAGR (Compounded Annual Growth Rate).
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    num_years = float(len(returns)) / ann_factor
    start_value = 100
    end_value = cum_returns(returns, starting_value=start_value).iloc[-1]
    total_return = (end_value - start_value) / start_value
    annual_return = (1. + total_return) ** (1. / num_years) - 1

    return annual_return


def annual_volatility(returns, period=DAILY, annualization=None):
    """
    Determines the annual volatility of a strategy.

    :param returns : pd.Series
        Periodic returns of the strategy, noncumulative.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Annual volatility.
    """
    if len(returns) < 2:
        return np.nan
    ann_factor = annualization_factor(period, annualization)
    volatility = returns.std() * (ann_factor ** (1.0 / 2.0))
    return volatility


def calmar_ratio(returns, period=DAILY, annualization=None):
    """
    Determines the Calmar ratio, or drawdown ratio, of a strategy.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Calmar ratio (drawdown ratio) as float. Returns np.nan if there is no
        calmar ratio.
    """

    max_dd = max_drawdown(returns=returns)
    if max_dd < 0:
        temp = annual_return(
            returns=returns,
            period=period,
            annualization=annualization
        ) / abs(max_dd)
    else:
        return np.nan

    if np.isinf(temp):
        return np.nan

    return temp


def omega_ratio(returns, risk_free=0.0, required_return=0.0,
                annualization=APPROX_BDAYS_PER_YEAR):
    """
    Determines the Omega ratio of a strategy.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param risk_free : int, float
        Constant risk-free return throughout the period
    :param required_return : float, optional
        Minimum acceptance return of the investor. Threshold over which to
        consider positive vs negative returns. It will be converted to a
        value appropriate for the period of the returns. E.g. An annual minimum
        acceptable return of 100 will translate to a minimum acceptable
        return of 0.018.
    :param annualization : int, optional
        Factor used to convert the required_return into a daily
        value. Enter 1 if no time period conversion is necessary.

    :return float
        Omega ratio.
    """

    if len(returns) < 2:
        return np.nan

    if annualization == 1:
        return_threshold = required_return
    elif required_return <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** \
            (1. / annualization) - 1

    returns_less_thresh = returns - risk_free - return_threshold

    numer = sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * sum(returns_less_thresh[returns_less_thresh < 0.0])

    if denom > 0.0:
        return numer / denom
    else:
        return np.nan


def sharpe_ratio(returns, risk_free=0, period=DAILY, annualization=None):
    """
    Determines the Sharpe ratio of a strategy.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param risk_free : int, float
        Constant risk-free return throughout the period.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Sharpe ratio.
        np.nan
        If insufficient length of returns or if if adjusted returns are 0.
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    returns_risk_adj = returns - risk_free

    if np.std(returns_risk_adj) == 0:
        return np.nan

    return np.mean(returns_risk_adj) / np.std(returns_risk_adj, ddof=1) * \
        np.sqrt(ann_factor)


def sortino_ratio(returns, required_return=0, period=DAILY,
                  annualization=None):
    """
    Determines the Sortino ratio of a strategy.

    :param returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
    :param required_return: float / series
        minimum acceptable return
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return depends on input type, series ==> float, DataFrame ==> pd.Series
        Annualized Sortino ratio.

    """

    if len(returns) < 2:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    if len(returns) < 2:
        return np.nan

    mu = np.nanmean(returns - required_return, axis=0)
    sortino = mu / downside_risk(returns, required_return)
    if len(returns.shape) == 2:
        sortino = pd.Series(sortino, index=returns.columns)
    return sortino * ann_factor


def downside_risk(returns, required_return=0, period=DAILY, annualization=None):
    """
    Determines the downside deviation below a threshold

    :param returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
    :param required_return: float / series
        minimum acceptable return
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return depends on input type, series ==> float, DataFrame ==> pd.Series
    Annualized downside deviation
    """

    if len(returns) < 1:
        return np.nan

    ann_factor = annualization_factor(period, annualization)

    downside_diff = returns - required_return
    mask = downside_diff > 0
    downside_diff[mask] = 0.0
    squares = np.square(downside_diff)
    mean_squares = np.nanmean(squares, axis=0)
    dside_risk = np.sqrt(mean_squares) * np.sqrt(ann_factor)
    if len(returns.shape) == 2:
        dside_risk = pd.Series(dside_risk, index=returns.columns)
    return dside_risk


def information_ratio(returns, bench_returns):
    """
    Determines the Information ratio of a strategy.

    :param returns : pd.Series or pd.DataFrame
        Daily returns of the strategy, noncumulative.
        - See full explanation in :func:`~qrisk.stats.cum_returns`.
    :param bench_returns: float / series
        Benchmark return to compare returns against.

    :return float
        The information ratio.
    """
    if len(returns) < 2:
        return np.nan

    active_return = returns - bench_returns
    tracking_error = np.std(active_return, ddof=1)
    if np.isnan(tracking_error):
        return 0.0
    if tracking_error == 0:
        return np.nan
    return np.mean(active_return) / tracking_error


def alpha_beta(returns, bench_returns, risk_free=0.0, period=DAILY,
               annualization=None):
    """
    Calculates annualized alpha and beta.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    :param risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.

    :return float
        Alpha, Beta
    """
    if len(returns) < 2:
        return np.nan, np.nan

    ann_factor = annualization_factor(period, annualization)

    y = (returns - risk_free).loc[bench_returns.index].dropna()
    x = (bench_returns - risk_free).loc[y.index].dropna()
    y = y.loc[x.index]
    beta, alpha = stats.linregress(x.values, y.values)[:2]
    return alpha * ann_factor, beta


def alpha(returns, bench_returns, risk_free=0.0, period=DAILY,
          annualization=None):
    """
    Calculates annualized alpha.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    :param risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.
    :param period : str, optional
        Defines the periodicity of the 'returns' data for purposes of
        annualizing. Value ignored if `annualization` parameter is specified.
        Defaults are:
            'monthly':12
            'weekly': 52
            'daily': 252
    :param annualization : int, optional
        Used to suppress default values available in `period` to convert
        returns into annual returns. Value should be the annual frequency of
        `returns`.
        - See full explanation in :func:`~qrisk.stats.annual_return`.

    :return float
        Alpha.
    """

    if len(returns) < 2:
        return np.nan

    return alpha_beta(returns,
                      bench_returns,
                      risk_free=risk_free,
                      period=period,
                      annualization=annualization)[0]


def beta(returns, bench_returns, risk_free=0.0):
    """
    Calculates beta.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    :param risk_free : int, float, optional
        Constant risk-free return throughout the period. For example, the
        interest rate on a three month us treasury bill.

    :return float
        Beta.
    """

    if len(returns) < 2:
        return np.nan

    return alpha_beta(returns, bench_returns, risk_free=risk_free)[1]


def stability_of_timeseries(returns):
    """
    Determines R-squared of a linear fit to the cumulative
    log returns. Computes an ordinary least squares linear fit,
    and returns R-squared.
    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.

    :return float
        R-squared.
    """
    if len(returns) < 2:
        return np.nan

    cum_log_returns = np.log1p(returns).cumsum()
    rhat = stats.linregress(np.arange(len(cum_log_returns)),
                            cum_log_returns.values)[2]

    return rhat ** 2


def tail_ratio(returns):
    """
    Determines the ratio between the right (95%) and left tail (5%).

    For example, a ratio of 0.25 means that losses are four times
    as bad as profits.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.

    :return float
        tail ratio
    """
    if len(returns) < 1:
        return np.nan

    return np.abs(np.percentile(returns, 95)) / \
        np.abs(np.percentile(returns, 5))


def win_ratio(returns, bench_returns):
    """
    Determine the ratio that what percentage of terms the strategy performs
    better than the benchmark.

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.
    :return: float
        win ratio
    """
    temp = (returns - bench_returns).dropna()
    return len(temp > 0)/float(len(temp))


def annual_active_return(returns, bench_returns):
    """
    Determine the annualized active return of the strategy compared to the benchmark

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    :return: float
        annualized active return
    """
    active_return = returns - bench_returns
    return annual_return(active_return, period=MONTHLY, annualization=None)


def max_min_active_return(returns, bench_returns):
    """
    Determine the maximum and minimum active return of the strategy compared to the benchmark
    within the strategy periods

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.
    :param bench_returns : pd.Series
        Daily noncumulative returns of the factor to which beta is
        computed. Usually a benchmark such as the market.
        - This is in the same style as returns.

    :return: float
        max and min active return
    """
    active_return = returns - bench_returns
    return active_return.min(), active_return.max()


def return_perf_metrics(returns, bench_returns=None):
    """
    Calculates various performance metrics of a strategy

    :param returns : pd.Series
        Daily returns of the strategy, noncumulative.

    :param bench_returns : pd.Series (optional)
        Daily noncumulative returns of the benchmark.
        - This is in the same style as returns.
        If None, do not compute alpha, beta, and information ratio.

    :return pd.Series
        Performance metrics.
    """
    return_metrics = pd.Series()

    for stat_func in SIMPLE_STAT_FUNCS:
        return_metrics[stat_func.__name__] = stat_func(returns)

    if bench_returns is not None:
        for stat_func in BENCH_STAT_FUNCS:
            return_metrics[stat_func.__name__] = stat_func(returns, bench_returns)
    return return_metrics


SIMPLE_STAT_FUNCS = [
    annual_return,
    annual_volatility,
    sharpe_ratio,
    calmar_ratio,
    downside_risk,
    stability_of_timeseries,
    max_drawdown,
    omega_ratio,
    sortino_ratio,
    stats.skew,
    stats.kurtosis,
    tail_ratio
]

BENCH_STAT_FUNCS = [
    information_ratio,
    alpha,
    beta,
    win_ratio,
    annual_active_return,
    max_min_active_return
]


# IC Analysis Metrics
def information_coefficient(factors, returns, cov=None, method='normal'):
    """
    :param factors: DataFrame or Series, current term factor data of each stock
    :param returns: DataFrame or Series, next term return data of each stock
    :param cov: (optional) numpy.array, covirance matrix of stocks
    :param method: str, default 'normal', or you can choose 'rank' or 'risk_adj'
    :return:
    """
    if returns is None:
        return (np.nan, np.nan)
    factors = factors.dropna() if factors.isnull().any() else factors
    returns = returns.dropna() if returns.isnull().any() else returns

    ret1, ret2 = factors.align(returns, join='inner')
    if method == 'normal':
        return stats.pearsonr(ret1, ret2)
    elif method == 'rank':
        return stats.spearmanr(ret1, ret2)
    elif method == 'risk_adj':
        return _risk_IC(ret1, ret2, cov)


def _risk_IC(df_fac, df_ret, cov):
    """
    风险调整信息系数
    cov协方差矩阵
    TODO: check error
    """
    n = len(df_fac)
    W = np.ones([n]) / n
    rf = 0.02
    R = df_ret.values
    target = lambda W: 1 / \
                       ((sum(W * R) - rf) / np.sqrt(
                           np.dot(np.dot(W, cov), W)))
    b = [(0., 1.) for i in range(n)]  # boundary condition
    c = ({'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # summation condition
    optimized = optimize.minimize(
        target, W, method='SLSQP', bounds=b, constraints=c)
    weights = optimized.x
    df_ret_w = df_ret * weights
    ret = stats.pearsonr(df_fac, df_ret_w)
    return list(ret)


