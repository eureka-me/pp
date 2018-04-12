#! env python
# -*- coding: utf-8 -*-

# import os
# import sys
from collections import defaultdict
import csv
import datetime as dt
from logging import getLogger
# from tqdm import tqdm


# ponpare.Tools
# Date: 2018/03/21
# Filename: Tools 

__author__ = 'takutohasegawa'
__date__ = "2018/03/21"


class Tools:
    def __init__(self, config):

        self.config = config
        self.coupon_list_train_file_path = config['GENERAL']['COUPON_LIST_TRAIN_FILE_PATH']
        self.coupon_list_test_file_path = config['GENERAL']['COUPON_LIST_TEST_FILE_PATH']
        self.coupon_list_file_paths = [self.coupon_list_train_file_path, self.coupon_list_test_file_path]
        self.coupon_detail_train_file_path = config['GENERAL']['COUPON_DETAIL_TRAIN_FILE_PATH']

        self.user_list_file_path = config['GENERAL']['USER_LIST_FILE_PATH']
        self.logger = getLogger(__name__)

    def get_coupon_dic(self, test=False):
        """クーポン情報辞書, クーポンリストの各列の水準リストの取得"""

        self.logger.debug('START: Tools.get_coupon_dic')

        dic, dic2 = defaultdict(dict), defaultdict(list)
        paths = [self.coupon_list_test_file_path] if test else self.coupon_list_file_paths
        for coupon_list_file_path in paths:
            with open(coupon_list_file_path, 'r', encoding='UTF-8') as f:

                reader = csv.reader(f)
                header = next(reader)
                ix = {h: i for i, h in enumerate(header)}
                cols = [c for c in header if c != 'COUPON_ID_hash']

                for row in reader:
                    cid = row[ix['COUPON_ID_hash']]
                    for c in cols:
                        dic[cid][c] = row[ix[c]]

                        if row[ix[c]] not in dic2[c]:
                            dic2[c].append(row[ix[c]])

        return dic, dic2

    def get_available_coupon_id_dic(self, test=False):
        """その週に購入可能であったクーポンIDリストを返す辞書を取得する"""

        self.logger.debug('START: Tools.get_available_coupon_id_dic')

        coupon_dic, _ = self.get_coupon_dic(test)
        del _

        dic = defaultdict(list)
        for cid, _dic in coupon_dic.items():

            available_week_stdates = self.get_available_week_stdates(self.get_stdate(_dic['DISPFROM']),
                                                                     self.get_stdate(_dic['DISPEND']))
            for _date in available_week_stdates:
                dic[_date].append(cid)

        return dic

    @staticmethod
    def get_available_week_stdates(st_date, en_date):

        date_list, _date = [], st_date
        while _date <= en_date:
            date_list.append(_date)
            _date += dt.timedelta(days=7)

        return date_list

    def get_coupon_purchase_user_dic(self):
        self.logger.debug('START: Tools.get_coupon_purchase_user_dic')

        dic = defaultdict(lambda: defaultdict(list))
        with open(self.coupon_detail_train_file_path, 'r', encoding='UTF-8') as f:

            reader = csv.reader(f)
            header = next(reader)
            ix = {h: i for i, h in enumerate(header)}

            for row in reader:
                _date = self.get_stdate(row[ix['I_DATE']])
                dic[row[ix['COUPON_ID_hash']]][_date].append(row[ix['USER_ID_hash']])

        return dic

    def get_user_purchase_date_cid_dic(self):
        self.logger.debug('START: Tools.get_user_purchase_date_cid_dic')

        dic = defaultdict(list)
        with open(self.coupon_detail_train_file_path, 'r', encoding='UTF-8') as f:

            reader = csv.reader(f)
            header = next(reader)
            ix = {h: i for i, h in enumerate(header)}

            for row in reader:
                _date = self.get_stdate(row[ix['I_DATE']])
                cid = row[ix['COUPON_ID_hash']]
                dic[row[ix['USER_ID_hash']]].append((_date, cid))

        return dic

    @staticmethod
    def get_stdate(_datetime):

        _date = dt.datetime.strptime(_datetime[:10], '%Y-%m-%d')
        stdate = _date - dt.timedelta(days=(_date.weekday() + 1) % 7)

        return stdate

    def get_user_profile_dic(self):
        self.logger.debug('START: Tools.get_user_profile_dic')

        dic = defaultdict(dict)
        with open(self.user_list_file_path, 'r', encoding='UTF-8') as f:

            reader = csv.reader(f)
            header = next(reader)
            ix = {h: i for i, h in enumerate(header)}

            for row in reader:
                uid = row[ix['USER_ID_hash']]

                for c in header:
                    if c == 'USER_ID_hash':
                        continue
                    elif c == 'REG_DATE' or c == 'WITHDRAW_DATE':
                        dic[uid][c] = self.get_stdate(row[ix[c]]) if row[ix[c]] != 'NA' else None
                    else:
                        dic[uid][c] = row[ix[c]]

        return dic


if __name__ == '__main__':

    tl = Tools()
    # tl.get_coupon_dic()
    tl.get_user_profile_dic()
