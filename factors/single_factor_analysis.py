# coding: utf8
from factors.data_type import FactorData
from factors.util import get_factor_name


def single_factor_analysis(data, pipeline, params):
    """
    单因子分析
    Args:
        params:
        data (DataFrame): 一个multi_index 数据框, 有因子, 对应下期收益率, 市值列. multi_index-->level0=dt, level1=code
        pipeline(List): 前面N-1个为对数据进行处理, 最后一个元素为一个元组,元组的元素为xxx_analysis

    Returns:
        单因子分析结果
    Examples:
        from analysis import filter_out_st, filter_out_suspend, filter_out_recently_ipo
        from analysis import prepare_data, add_group, de_extreme
        from analysis import information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis

        data = prepare_data(factor_name="M004009", index_code='000300', benchmark_code='000300',
        start_date='2015-01-01', end_date='2016-01-01, freq='M')
        params = {'de_extreme': {'num':1, 'method': 'mad'},
                  'standardize': dict(method='cap'),
        }
        pipeline = [filter_out_st, filter_out_suspend, filter_out_recently_ipo, add_group, de_extreme,
        (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
        result = single_factor_analysis(pipeline)
    """
    X = data.copy()
    for func in pipeline[:-1]:
        X = func(X, **(params.get(func.__name__, {})))

    result_dict = {}
    for func in pipeline[-1]:
        result_dict[func.__name__] = func(X, **(params.get(func.__name__, {})))

    factor_name = get_factor_name(data)
    factor_result = FactorData(name=factor_name, **result_dict)
    return factor_result
