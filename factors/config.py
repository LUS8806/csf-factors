# -*- coding: utf-8 -*-
import os

pwd = os.path.dirname(os.path.abspath(__file__))
TRADE_CAL_PATH = os.path.join(pwd, 'trade_cal.csv')
FACTORS_DETAIL_PATH = os.path.join(pwd, 'quant_dict_factors_all.csv')
END_TD_PATH = os.path.join(pwd, 'end_td.txt')
EX_METHOD = 'mad'  #　标准化std, mad
SCALE_METHOD = 'normal'  # normal, cap, sector

STOCK_FILTER = {
    'ST': True,     # ST股票
    'TP': True,     # 停牌的股票
    'SLD': 60,      # 上市未满60天
    'FNAN': True    # 因子值为空的股票
}