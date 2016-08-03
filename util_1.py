# -*- coding: UTF-8 -*-

"""
功能性函数
"""
import datetime
import math
import time
from collections import OrderedDict, MutableMapping
from functools import wraps

import numpy as np
import pandas as pd
from pymongo import MongoClient
from scipy.stats import pearsonr, spearmanr
from sqlalchemy import create_engine
from open_alpha.config import DEFALUT_HOST

import logging
import re

logger = logging.getLogger(__name__)
logging.basicConfig()
logger.setLevel(logging.DEBUG)


def m_end_td(date_index):
    """
    生成每个月最后一个交易日
    """
    month_end_list = []
    ret = OrderedDict()
    for dt in date_index:
        temp_y = dt.year
        temp_m = dt.month
        temp_d = dt.day
        if temp_y in ret.keys():
            if temp_m in ret[temp_y].keys():
                if temp_d > ret[temp_y][temp_m]:
                    ret[temp_y][temp_m] = temp_d
            else:
                ret[temp_y][temp_m] = temp_d
        else:
            ret[temp_y] = OrderedDict()
            ret[temp_y][temp_m] = temp_d
    for y in ret.keys():
        for m in ret[y].keys():
            month_end_list.append(datetime.datetime(y, m, ret[y][m]))
    return month_end_list


def w_end_td(date_index):
    """
    生成每个周最后一个交易日
    """
    week_end_list = []
    n = len(date_index)
    for i in range(0, n - 1):
        w0 = datetime.datetime.weekday(date_index[i])
        dif = date_index[i + 1] - date_index[i]
        if w0 == 4:
            week_end_list.append(date_index[i])
        elif w0 == 3 and dif >= datetime.timedelta(4):
            week_end_list.append(date_index[i])
        elif w0 == 2 and dif >= datetime.timedelta(5):
            week_end_list.append(date_index[i])
        elif w0 == 1 and dif >= datetime.timedelta(6):
            week_end_list.append(date_index[i])
        elif w0 == 0 and dif >= datetime.timedelta(7):
            week_end_list.append(date_index[i])
        else:
            pass
        week_end_list.append(date_index[-1])
        return week_end_list


def q_end_td(date_index):
    """
    生成每个季度最后一个交易日
    """
    q_end_list = []
    temp_list = m_end_td(date_index)
    for dt in temp_list:
        if dt.month in [3, 6, 9, 12]:
            q_end_list.append(dt)
    return q_end_list


# 获取股票相关信息
# MOVE TO get_data
def get_index_components(idx_code, date):
    """
    获取指定时间SAM指数及常用指数成份股代码
    @输入:
    idx_code: str，指数代码
    date: 日期，'2015-08-31'
    @返回：
    components: list, 成分股代码列表
    """
    # date = date if date else str(datetime.datetime.today().date())
    # print date
    conn = set_mongo_cond(DEFALUT_HOST)
    sam_tb = conn.ada.index_specimen_stock
    com_tb = conn.metrics.idx_stock_history
    if idx_code[0] in list('0123456789'):
        alrec = com_tb.find(
            {"idxcd": idx_code, "st": {"$lte": date}, "et": {"$gte": date}}, {
                '_id': 0, "secus": 1, "st": 1})
        components = list(alrec)[0]['secus']
        components = list(set(components) - set(get_not_trade_stock(date)))  # 剔除停牌股\st股票
    else:
        date = date.replace('-', '')
        alrec = sam_tb.find(
            {"idxcd": idx_code, "st": {"$lte": date}, "et": {"$gte": date}}, {
                '_id': 0, "secus": 1, "st": 1})
        components = []
        rec_lst = list(alrec)
        if rec_lst:
            components = [i['secu'] for i in rec_lst[0]['secus']]
    return components

# MOVE TO get_data
def get_not_trade_stock(date):
    '''
    jyzt!="N"   交易状态 N通常状态；
    zqjb !="N"   证券级别  N 表示正常状态
    tpbz == "F"  停牌标志   T-停牌
    ## engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.21/ada-fd')
    ## sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    '''
    engine = create_engine('mysql://pd_team:pd_team321@!@122.144.134.95/ada-fd')
    sql_trade = (""" SELECT * FROM `ada-fd`.hq_stock_trade where dt = '{}' """).format(date)
    trade = pd.read_sql_query(sql_trade, engine).set_index('tick')
    st = trade.query(' zqjb == "s" or zqjb == "*" ').index.tolist()
    sql_tp = (""" SELECT tick FROM `ada-fd`.hq_stock_tp where dt = '{}' """).format(date)
    tingpai = pd.read_sql_query(sql_tp, engine)['tick'].tolist()
    __stocks=list(set(st) | set(tingpai))
    _stocks = [x for x in __stocks if x.startswith('0') or x.startswith('3') or x.startswith('6')]
    stocks = [str(x) + ('_SH_EQ') if x.startswith('6') else str(x) + ('_SZ_EQ') for x in _stocks]
    return stocks


# MOVE TO get_data
def get_stock_lst_date(codes):
    """
    查询股票上市日期, 用来过滤数据
    输入：codes，单个股票代码或股票代码列表
    返回：{'code': 'dt'} 字典格式
    """
    if not isinstance(codes, list):
        codes = [codes]
    _, _, tb_base_stock = set_mongo_cond(DEFALUT_HOST, 'ada', 'base_stock')
    alrec = tb_base_stock.find(
        {"code": {"$in": codes}}, {"_id": 0, "code": 1, "ls.dt": 1})
    dt_lst = list(alrec)
    ret = {}
    for i in dt_lst:
        ret[i['code']] = i['ls']['dt']
    return ret


# MOVE TO get_data
def get_stock_cap(codes, start_date=None, end_date=None):
    """
    用于查询股票流通市值
    输入参数：
    codes: 单个股票代码或股票代码列表
    start_date: 开始日期
    end_date: 结束日期
    """
    if DEFALUT_HOST == '192.168.0.222':
        _, _, tb = set_mongo_cond(DEFALUT_HOST, db_name='metrics_test',
                                  tb_name='comm_idx_price')
    else:
        _, _, tb = set_mongo_cond(db_name='metrics', tb_name='comm_idx_price')

    if not isinstance(codes, list):
        codes = [codes]
    alrec = tb.find({}, {})
    pass


# 数据库操作相关

# Replace by csf function
def get_close_ndays_back(stocks, date, days):
    """
    获取前n天的股价
    """
    if isinstance(stocks, list):
        stocks = [stocks]
    if len(stocks[0]) > 6:
        stocks = [i[0:6] for i in stocks]
    engine = ut.set_sql_con()
    sql = "select dt, tick, close from hq_price where dt <=%s and tick in %s " % (
        date, tuple(stocks))
    get_sql = pd.read_sql(engine, sql)
    get_sql['tick'] = get_sql['tick'].map(
        lambda x: x + '_SH_EQ' if x[0] == '6' else x + 'SZ_EQ')
    ar = get_sql.ix[:, ['dt', 'tick']].T.values
    mul_idx = pd.MultiIndex.from_arrays(ar)
    ts = pd.Series(index=mul_idx, data=get_sql.close.values)
    df_pr = ts.unstack()
    return df_pr


# 数据结构操作

def flatten(d, parent_key='', sep='.'):
    '''
    字典flatten
    '''
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def write_to_sql():
    '''
    往sql写数
    '''
    pass


def write_to_mongo(args, mongo_host, db_name, tb_name, mode='insert'):
    '''
    往mongo写数据
    '''
    _, _, tb = set_mongo_cond(mongo_host, db_name, tb_name)

    if mode == 'insert':
        tb.insert(args)


def set_mongo_cond(host='122.144.134.95', db_name=None, tb_name=None):
    """
    mongo连接设置
    host: ip_address
    db_name: database name
    tb_name: table name
    """
    conn = MongoClient(host, 27017)
    ret = [conn]
    if db_name:
        db = conn[db_name]
        ret.append(db)
        if tb_name:
            tb = db[tb_name]
            ret.append(tb)
    ret = tuple(ret)
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def set_sql_con(sql_addr=None):
    """
    sql连接设置
    sql_addr: 'mysql://ada_user:ada_user@122.144.134.3/ada-fd'
    """
    sql_default = 'mysql://ada_user:ada_user@122.144.134.3/ada-fd'
    engine = create_engine(sql_addr if sql_addr else sql_default)
    return engine


def timethis(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        print('{} cost {} seconds'.format(func.__name__, t1 - t0))
        return ret

    return wrapper


def form_mongo_to_df(tb_name, pos, filters):
    """
    输入mongo表连接，查询字段名(全字段，如quant.f18)，过滤条件
    tb_conn: mongo表连接
    pos: 查找的字段名
    filters：过滤条件，字典格式
    返回DataFrame格式
    """
    if DEFALUT_HOST == '192.168.250.200':
        _, db_metrics = set_mongo_cond(host=DEFALUT_HOST,
                                       db_name='metrics')
    else:
        _, db_metrics = set_mongo_cond(db_name='metrics')

    tb = {
        'pr': db_metrics.comm_idx_price_his_a,
        'ltm': db_metrics.comm_idx_quant_his_a,
        'ytd': db_metrics.comm_idx_quant_ytd_his_a,
        'tech': db_metrics.comm_idx_tech_his_a
    }

    projection_dict = {}
    for k in pos:
        key = k.replace('.', '#')
        val = ''.join(["$", k])
        projection_dict[key] = val
    projection_direct = {
        'pr': dict(_id=False, dt=True, secu=True, **projection_dict),
        'ltm': dict(_id=False, y=True, secu=True, **projection_dict),
        'ytd': dict(_id=False, y=True, secu=True, **projection_dict),
        'tech': dict(_id=False, dt=True, secu=True, **projection_dict)
    }

    pipeline = [
        {"$match": filters},
        {"$project": projection_direct[tb_name]}
    ]

    all_rec = tb[tb_name].aggregate(pipeline)

    df = pd.DataFrame(list(all_rec))
    df.columns = [col.replace("#", ".") for col in df.columns]

    return df



# MOVE TO metrics
class InfoCof(object):
    """
    信息系数，三种算法：
    普通信息系数：因子暴露值与收益率的相关系数
    排序信息系数：因子暴露值排序与收益率排序的信息系数
    风险调整信息系数：经均值方差确定权重后的信息系数
    """

    def __init__(self, df_fac=None, df_ret=None, cov=None, method='normal'):
        self.fac = df_fac
        self.ret = df_ret
        self.cov = cov

    @staticmethod
    def _normal_IC(df_fac, df_ret):
        """
        信息系数计算
        """
        ret = pearsonr(df_fac, df_ret)
        return list(ret)

    @staticmethod
    def _rank_IC(df_fac, df_ret):
        """
        排序信息系数
        """
        ret = spearmanr(df_fac, df_ret)
        return list(ret)

    @staticmethod
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
                           ((sum(W * R) - rf) / math.sqrt(
                               np.dot(np.dot(W, cov), W)))
        b = [(0., 1.) for i in range(n)]  # boundary condition
        c = (
            {'type': 'eq', 'fun': lambda W: sum(W) - 1.})  # summation condition
        optimized = scipy.optimize.minimize(
            target, W, method='SLSQP', bounds=b, constraints=c)
        weights = optimized.x
        df_ret_w = df_ret * weights
        ret = pearsonr(df_fac, df_ret_w)
        return list(ret)

    def IC(self, df_fac=None, df_ret=None, cov=None, method='normal'):
        df_fac = df_fac if not df_fac.empty else self.fac
        df_ret = df_ret if not df_ret.empty else self.ret
        cov = cov if cov != None else self.cov
        try:
            df = pd.concat([df_fac, df_ret]).dropna()
        except Exception, e:
            logger.exception(e)
            raise
        if not df.empty:
            if method == 'normal':
                ret = self._normal_IC(df_fac, df_ret)
            elif method == 'rank':
                ret = self._rank_IC(df_fac, df_ret)
            elif method == 'risk':
                ret = self._risk_IC(df_fac, df_ret, cov)
        else:
            logger.critical(
                'critical error occured! df_fac:{}, df_ret{}'.format(
                    df_fac.empty, df_ret.empty))
            ret = [np.nan, np.nan]
        return ret
