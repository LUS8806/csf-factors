# coding: utf8
import pandas as pd

from factors.analysis import add_group
from factors.data_type import FactorData
from factors.metrics import information_coefficient


def score(fac_ret_data, method='equal_weighted', ascending=False, rank_method='first', na_option='keep',
          score_window=12,
          num_group=5, group_ascending=False):
    """
    给多因子打分
    Args:
        fac_ret_data (DataFrame): multi_indexed 数据框, 有若干因子列, ret, cap, benchmark_returns等.
        method(str): 打分方法.
                    {'equal_weighted', 'ic_weighted', 'icir_weighted', 'ret_weighted', 'ret_icir_weighted'}
                    * equal_weighted: 等权法
                    * ic_weighted: ic 加权
                    * icir_weighted: icir 加权
                    * ret_weighted: 收益率加权, 取第一个分组的平均收益率为该因子收益率.
                    * ret_icir_weighted: 因子icir加权

        ascending (bool or dict): 默认排序, 可以是一个bool值, 所有因子相同排序, 也可以是一个字典,其key是因子名称, value是bool值.
        rank_method (str) : {'average', 'min', 'max', 'first', 'dense'}
                            * average: average rank of group
                            * min: lowest rank in group
                            * max: highest rank in group
                            * first: ranks assigned in order they appear in the array
                            * dense: like 'min', but rank always increases by 1 between groups
        na_option (str): {'keep', 'top', 'bottom'}
                            * keep: leave NA values where they are
                            * top: smallest rank if ascending
                            * bottom: smallest rank if descending
        score_window (int): 滑动窗口大小, 对equal_weighted无效
        num_group(int): 分组数, 该参数仅对method='ret_weighted' 有效
        group_ascending(bool or dict): 分组时的排序, 默认值False,所有因子分组时按降序排序.
                                       也可指定具体某一个或多个因子为升序,其他默认降序.
                                       该参数仅对method='ret_weighted' 有效

    Returns:
        DataFrame, 打完分的DataFrame,
    Raises:
        KeyError, 如果输入的打分方法有误,会引发该错误.

    """
    factor_names = set(fac_ret_data.columns) - {'ret', 'cap', 'benchmark_returns', 'group'}
    # 非因子列名称
    other_names = sorted(set(fac_ret_data.columns) - factor_names)
    if len(factor_names) >= 1:
        factor_names = sorted(factor_names)
    else:
        raise ValueError('fac_ret_data must have at least 1 factor, got 0.')

    def get_ic():
        ic = fac_ret_data.groupby(level=0).apply(
            lambda frame: [information_coefficient(frame[fac], frame['ret'])[0] for fac in factor_names])
        # 上一步骤得到的结果是一个序列,序列元素是一列表,需要将其拆开
        ic = pd.DataFrame(ic.tolist(), index=ic.index, columns=factor_names)
        return ic

    def get_rank():
        """
        取得因子排名. 规则如下: ascending 默认为False, 也就是值越大其rank越小(与成绩类似).会被分入靠前的组.
        对于那些需要升序排列的因子, 我们将其乘以-1后, 统一用降序.
        对于NA的处理,是放到最后,强制使其rank为0
        Args:

        Returns:
            DataFrame: 因子rank
        """
        if isinstance(ascending, bool):
            rnk = fac_ret_data[factor_names].groupby(level=0).rank(ascending=ascending, na_option=na_option,
                                                                   method=rank_method)
        elif isinstance(ascending, dict):
            default_ascending = dict(zip(factor_names, [False] * len(factor_names)))
            if len(ascending) != len(factor_names):
                print('ascending 长度与因子数目不同, 未指明的将按照默认降序排序(大值排名靠前).')
            default_ascending.update(ascending)
            rnk_list = [fac_ret_data[col].groupby(level=0).rank(asending=default_ascending[col], na_option=na_option,
                                                                method=rank_method)
                        for col in factor_names]
            rnk = pd.concat(rnk_list, axis=1)
        # 假设有k个NA, 未执行下句时, rank 值 从1..(N-k), 执行后, rnk值是从k+1..N
        rnk += rnk.isnull().sum()
        # fillna后, NA的rank被置为0.
        rnk = rnk.fillna(0.0)
        return rnk

    def equal_weighted():
        """
        因子打分:等权法
        Returns:
            DataFrame: 一个数据框,有score列,和ret, cap,等列
        """
        rnk = get_rank()
        score_ = fac_ret_data[factor_names].mul(rnk.mean(axis=1), axis=0).sum(axis=1).to_frame().rename(
            columns={0: 'score'})

        return score_.join(fac_ret_data[other_names])

    def ic_weighted():
        ic = get_ic().abs()
        rolling_ic = ic.rolling(score_window, min_periods=1).mean()
        weight = rolling_ic / rolling_ic.sum(axis=1)

        rank = get_rank()
        score_ = (rank * weight).sum(axis=1).to_frame().rename(columns={0: 'score'})

        return score_.join(fac_ret_data[other_names])

    def icir_weighted():
        ic = get_ic().abs()
        rolling_ic = ic.rolling(score_window, min_periods=1).mean()
        rolling_std = ic.rolling(score_window, min_periods=1).std()
        weight = rolling_ic / rolling_std
        weight.loc[0, :] = rolling_ic.loc[0, :]
        rank = get_rank()
        score_ = (rank * weight).sum(axis=1).to_frame().rename(columns={0: 'score'})
        return score_.join(fac_ret_data[other_names])

    def ret_weighted():
        first_group_returns = []
        group_ascending_default = dict(zip(factor_names, [False] * len(factor_names)))
        if isinstance(group_ascending, dict):
            group_ascending_default.update(**group_ascending)
        for factor_name in factor_names:
            data = fac_ret_data[[factor_name] + other_names]  # perhaps only ['ret'] other than other_names
            data = add_group(data, num_group=num_group, ascending=group_ascending_default[factor_name], method='first',
                             na_option='keep')
            # 取得每个因子第一个分组的收益率
            ret = data.groupby([data.index.get_level_values(0), data.group])['ret'].mean().unstack()[['Q01']].rename(
                columns={'Q01': factor_name})
        returns = pd.concat(first_group_returns, axis=1)
        returns = returns.rolling(12, min_periods=1).mean()
        weight = returns / returns.sum(axis=1)
        rank = get_rank()

        score_ = (rank * weight).sum(axis=1).to_frame().rename(columns={0: 'score'})

        return score_.join(fac_ret_data[other_names])

    valid_method = {'equal_weighted': equal_weighted,
                    'ic_weighted': ic_weighted,
                    'icir_weighted': icir_weighted,
                    'ret_weighted': ret_weighted}
    try:
        return valid_method[method]()
    except KeyError:
        print('{} is not a valid method. valid methods are: {}'.format(method, valid_method.keys()))
        raise


def multiple_factors_analysis(data, pipeline, params=None):
    """
    多因子分析
    Args:
        data (DataFrame): 一个multi_index 数据框, 有因子, 对应下期收益率, 市值列. multi_index-->level0=dt, level1=code
        pipeline(List): 前面N-1个为对数据进行处理, 最后一个元素为一个元组,元组的元素为xxx_analysis
        params (dict): key为pipeline里面的函数名称, value该函数的参数, 为一字典

    Returns:
        多因子分析结果
    Examples:
        from analysis import filter_out_st, filter_out_suspend, filter_out_recently_ipo
        from analysis import prepare_data, add_group, de_extreme
        from analysis import information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis

        data = prepare_data(factor_name=["M004009", "M004023"], index_code='000300', benchmark_code='000300',
        start_date='2015-01-01', end_date='2016-01-01, freq='M')
        pipeline = [filter_out_st, filter_out_suspend, filter_out_recently_ipo,de_extreme, standardize, score, add_group,
        (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
        params = {'de_extreme': {'num':1, 'method': 'mad'},
                  'standardize': dict(method='cap'),
        }
        result = single_factor_analysis(pipeline, params)
    """
    X = data.copy()
    for func in pipeline[:-1]:
        X = func(X, **(params.get(func.__name__, {})))

    result_dict = {}
    for func in pipeline[-1]:
        result_dict[func.__name__] = func(X, **(params.get(func.__name__, {})))

    # factor_name = get_factor_name(data)
    factor_result = FactorData(name='multiple', **result_dict)
    return factor_result
