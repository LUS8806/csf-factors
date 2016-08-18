from factors.analysis import filter_out_recently_ipo, filter_out_suspend, filter_out_st, return_analysis, standardize
from factors.analysis import information_coefficient_analysis, code_analysis, turnover_analysis
from factors.analysis import prepare_data, add_group, de_extreme
from factors.single_factor_analysis import single_factor_analysis
from factors.multiple_factor_analysis import score, multiple_factors_analysis


def test_multiple_factors_analysis():
    import os
    import pandas as pd
    data_path = 'data.csv'
    if not os.path.exists(data_path):
        data = prepare_data(factor_name=["M004009Y", 'M008005'],
                            index_code='000300',
                            benchmark_code='000300',
                            start_date='2013-01-01',
                            end_date='2016-01-01', freq='M')
        data.to_csv(data_path)
    else:
        data = pd.read_csv(data_path, dtype={'code': object}).set_index(['date', 'code'])
    pipeline = [filter_out_st, filter_out_suspend, filter_out_recently_ipo, de_extreme, standardize,
                score,
                add_group,
                (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
    params = {'de_extreme': {'num': 1, 'method': 'mad'},
              'standardize': dict(method='cap'),
              }
    result = multiple_factors_analysis(data, pipeline, params)

    print('done')

if __name__ == '__main__':
    test_multiple_factors_analysis()
