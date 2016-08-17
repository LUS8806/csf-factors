# coding: utf8
import pandas as pd

from factors.metrics import information_coefficient


def score(fac_ret_data, method='equal_weighted', asending=False, rank_method='first', na_option='keep',
          score_window=12):
    """
    给多因子打分
    Args:
        fac_ret_data (DataFrame): multi_indexed 数据框, 有若干因子列, ret, cap, benchmark_returns等.
        method(str): 打分方法, 等权法:equal_weighted, 市值加权:cap_weighted
        asending (bool or dict): 默认排序, 可以是一个bool值, 所有因子相同排序, 也可以是一个字典,其key是因子名称, value是bool值.
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



    Returns:
        DataFrame, 打完分的DataFrame

    """
    facotr_names = set(fac_ret_data.columns) - {'ret', 'cap', 'benchmark_returns', 'group'}
    # 非因子列名称
    other_names = sorted(set(fac_ret_data.columns) - facotr_names)
    if len(facotr_names) >= 1:
        facotr_names = sorted(facotr_names)
    else:
        raise ValueError('fac_ret_data must have at least 1 factor, got 0.')

    def get_ic():
        ic = fac_ret_data.groupby(level=0).apply(
            lambda frame: [information_coefficient(frame[fac], frame['ret'])[0] for fac in facotr_names])
        # 上一步骤得到的结果是一个序列,序列元素是一列表,需要将其拆开
        ic = pd.DataFrame(ic.tolist(), index=ic.index, columns=facotr_names)
        return ic

    def get_rank():
        """
        取得因子排名. 规则如下: ascending 默认为False, 也就是值越大其rank越小(与成绩类似).会被分入靠前的组.
        对于那些需要升序排列的因子, 我们将其乘以-1后, 统一用降序.
        对于NA的处理,是放到最后,强制使其rank为0
        Args:

        Returns:

        """
        if isinstance(asending, bool):
            rnk = fac_ret_data[facotr_names].groupby(level=0).rank(ascending=asending, na_option=na_option,
                                                                   method=rank_method)
        elif isinstance(asending, dict):
            default_ascending = dict(zip(facotr_names, [False] * len(facotr_names)))
            if len(asending) != len(facotr_names):
                print('ascending 长度与因子数目不同, 未指明的将按照默认降序排序(大值排名靠前).')
            default_ascending.update(asending)
            rnk_list = [fac_ret_data[col].groupby(level=0).rank(asending=default_ascending[col], na_option=na_option,
                                                                method=rank_method)
                        for col in facotr_names]
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
        score_ = fac_ret_data[facotr_names].mul(rnk.mean(axis=1), axis=0).sum(axis=1).to_frame().rename(
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
        raise NotImplementedError

    def ret_icir_weighted():
        raise NotImplementedError

    valid_method = {'equal_weighted': equal_weighted,
                    'ic_weighted': ic_weighted,
                    'icir_weighted': icir_weighted,
                    'ret_weighted': ret_weighted,
                    'ret_icir_weighted': ret_icir_weighted}
    try:
        return valid_method[method]()
    except KeyError:
        print('{} is not a valid method. valid methods are: {}'.format(method, valid_method.keys()))
        raise
