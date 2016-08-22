![logo](./logo.png)
# 数库factors平台

## 简介
factors平台是一个开源的多因子量化平台. 包含单因子分析,多因子分析与事件驱动分析.
如果您不喜欢写代码, 可以在数库上注册一个账户,

## 特色:
1. 容易使用: 本项目将多因子分析抽象成一个管道(pipeline).原始数据通过这些管道, 最终生成
多因子分析的报告.
![data pipeline](./multi-factor-analysis.svg)


## 安装

pip install factors

## Usage

```python
>>> from factors.analysis import (filter_out_recently_ipo,
                                filter_out_suspend,
                                filter_out_st,
                                return_analysis,
                                standardize)
>>> from factors.analysis import (information_coefficient_analysis,
                              code_analysis,
                              turnover_analysis)
>>> from factors.analysis import prepare_data, add_group, de_extreme
>>> from factors.multiple_factor_analysis import score, multiple_factors_analysis

>>> data = prepare_data(factor_name=["M004009Y", 'M008005'],
                    index_code='000300',
                    benchmark_code='000300',
                    start_date='2013-01-01',
                    end_date='2016-01-01', freq='M')

>>> pipeline = [filter_out_st, filter_out_suspend, filter_out_recently_ipo, de_extreme, standardize,
            score,
            add_group,
            (information_coefficient_analysis, return_analysis, code_analysis, turnover_analysis)]
>>> params = {'de_extreme': {'num': 1, 'method': 'mad'},
          'standardize': dict(method='cap'),
          'return_analysis': dict(plot=True),
          }
>>> result = multiple_factors_analysis(data, pipeline, params)
```
