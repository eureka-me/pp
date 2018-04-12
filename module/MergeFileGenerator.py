#! env python
# -*- coding: utf-8 -*-

import csv, random
import datetime as dt
from logging import getLogger
from collections import defaultdict
from module import Tools as tls

# ponpare.MergeFileGenerator
# Date: 2018/03/29
# Filename: MergeFileGenerator 

__author__ = 'takutohasegawa'
__date__ = "2018/03/29"


class MargeFileGenerator:
    def __init__(self, config, output_dir, date_from=dt.datetime(2010, 1, 1), date_until=dt.datetime(2020, 1, 1),
                 mode='train', purchase_count_thr=3, not_buy_sampling=10):

        self.config = config
        self.logger = getLogger(__name__)
        self.Tools = tls.Tools(config)
        self.mode = mode
        self.output_dir = output_dir
        self.purchase_count_thr = purchase_count_thr
        self.not_buy_sampling = not_buy_sampling

        # 生成時に実行
        # クーポン情報の取得
        test = True if mode == 'test' else False
        self.coupon_dic, self.label_dic = self.Tools.get_coupon_dic(test=test)

        # 購入可能なクーポンリストの取得
        self.available_coupon_id_dic = self.Tools.get_available_coupon_id_dic(test=test)

        # クーポン購入者辞書の取得
        self.coupon_purchase_user_dic = self.Tools.get_coupon_purchase_user_dic()

        # ユーザーのクーポン購入数の取得
        self.user_purchase_date_cid_dic = self.Tools.get_user_purchase_date_cid_dic()

        # ユーザー属性情報の取得
        self.user_profile_dic = self.Tools.get_user_profile_dic()

        # 週_cid リストの生成
        _date, self.date_cid_list = max(dt.datetime(2011, 7, 10), date_from), []
        while _date <= min(dt.datetime(2012, 6, 24), date_until):
            available_coupons = self.available_coupon_id_dic[_date]
            for cid in available_coupons:
                self.date_cid_list.append((_date, cid))
            _date += dt.timedelta(days=7)

        # ヘッダーの作成
        self.cols_user = ['SEX_ID', 'AGE', 'PREF_NAME']
        self.genre_labels = ['all'] + self.label_dic['GENRE_NAME']
        self.cols_coupon = ['CAPSULE_TEXT', 'GENRE_NAME', 'PRICE_RATE', 'CATALOG_PRICE', 'DISCOUNT_PRICE', 'DISPPERIOD',
                            'VALIDPERIOD', 'USABLE_DATE_MON', 'USABLE_DATE_TUE', 'USABLE_DATE_WED', 'USABLE_DATE_THU',
                            'USABLE_DATE_FRI', 'USABLE_DATE_SAT', 'USABLE_DATE_SUN', 'USABLE_DATE_HOLIDAY',
                            'USABLE_DATE_BEFORE_HOLIDAY', 'large_area_name', 'ken_name', 'small_area_name']

        self._header = ['uid', 'date', 'cid']
        self._header += self.cols_user
        self._header += ['purchase_' + genre + '_' + str(week) for genre in self.genre_labels
                         for week in [1, 2, 4, 13, 26]]
        self._header += ['visit_' + genre + '_' + str(week) for genre in self.genre_labels
                         for week in [1, 2, 4, 13, 26]]
        self._header += self.cols_coupon
        self._header += ['purchase']

    def generate_file(self, user_file):

        uid = user_file[:-4]

        _date_cid_list, output = self.get_date_cid_list(uid)

        if not output:
            self.logger.debug('SKIP: {}'.format(uid))
            return None

        # 出力ファイル
        fout = open(self.output_dir + user_file, 'w', encoding='SJIS')
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(self._header)

        purchase_history_dic = self.get_coupon_history(self.config['GENERAL']['PURCHASE_FILE_DIR'], user_file)
        visit_history_dic = self.get_coupon_history(self.config['GENERAL']['VISIT_FILE_DIR'], user_file)

        for _date, cid in _date_cid_list:

            _row = [uid, _date.strftime('%Y%m%d'), cid]

            # ユーザー属性情報の付与
            _row += [t[1] for t in self.user_profile_dic[uid].items() if t[0] in self.cols_user]

            # クーポン購買履歴情報の付与
            for genre in self.genre_labels:
                if genre in purchase_history_dic[_date]:
                    _row += [t[1] for t in sorted(purchase_history_dic[_date][genre].items(), key=lambda x: x[0])]
                else:
                    _row += [None if t[1] is None else 0 for t in sorted(purchase_history_dic[_date]['all'].items(),
                                                                         key=lambda x: x[0])]
            # クーポン閲覧履歴情報の付与
            for genre in self.genre_labels:
                if genre in visit_history_dic[_date]:
                    _row += [t[1] for t in sorted(visit_history_dic[_date][genre].items(), key=lambda x: x[0])]
                    # print(genre)
                    # print([t[1] for t in sorted(visit_history_dic[_date][genre].items(), key=lambda x: x[0])])
                else:
                    _row += [None if t[1] is None else 0 for t in sorted(visit_history_dic[_date]['all'].items(),
                                                                         key=lambda x: x[0])]
                    # print(genre)
                    # print([None if t[1] is None else 0 for t in sorted(visit_history_dic[_date]['all'].items(),
                    #                                                      key=lambda x: x[0])])

            # クーポン情報、クーポン購入有無情報の付与
            _row += [t[1] for t in self.coupon_dic[cid].items() if t[0] in self.cols_coupon]

            # 購買有無の付与
            _row += [1] if uid in self.coupon_purchase_user_dic[cid][_date] else [0]

            writer.writerow(_row)

        fout.close()

    def get_date_cid_list(self, uid):

        """
        クーポン購入有無情報から、アンダーサンプリング、レコードリストを返す
        mode: 'train':学習データ用、'test':テストデータ用
        """

        _date_cid_list = []
        if self.mode == 'train':

            # アクティブユーザーの判定の判定
            if len(self.user_purchase_date_cid_dic[uid]) < self.purchase_count_thr:
                return _date_cid_list, False

            # レコードのサンプリング
            _date_cid_not_buy, reg_date, withdraw_date = [], self.user_profile_dic[uid]['REG_DATE'], \
             self.user_profile_dic[uid]['WITHDRAW_DATE'] if self.user_profile_dic[uid]['WITHDRAW_DATE'] is not None \
                                                             else dt.datetime(2020, 1, 1)

            for _date_cid in self.date_cid_list:
                if (_date_cid[0] < reg_date) or (withdraw_date < _date_cid[0]):
                    continue
                if _date_cid not in self.user_purchase_date_cid_dic[uid]:
                    _date_cid_not_buy.append(_date_cid)

            if self.not_buy_sampling > 0:
                random.seed(0)
                _date_cid_list = self.user_purchase_date_cid_dic[uid] + \
                                 random.sample(_date_cid_not_buy,
                                               self.not_buy_sampling * len(self.user_purchase_date_cid_dic[uid]))
            else:
                _date_cid_list = self.user_purchase_date_cid_dic[uid] + _date_cid_not_buy

            _date_cid_list = [_date_cid for _date_cid in _date_cid_list if _date_cid[0] >= dt.datetime(2011, 7, 10)]
            _date_cid_list.sort(key=lambda x: x[0])

            return _date_cid_list, True

        elif self.mode == 'test':
            _date_cid_list, reg_date, withdraw_date = [], self.user_profile_dic[uid]['REG_DATE'], \
             self.user_profile_dic[uid]['WITHDRAW_DATE'] if self.user_profile_dic[uid]['WITHDRAW_DATE'] is not None \
                                                         else dt.datetime(2020, 1, 1)
            for _date_cid in self.date_cid_list:
                if (_date_cid[0] < reg_date) or (withdraw_date < _date_cid[0]):
                    continue
                if _date_cid not in self.user_purchase_date_cid_dic[uid]:
                    _date_cid_list.append(_date_cid)

            if len(_date_cid_list) > 0:
                return _date_cid_list, True
            else:
                return _date_cid_list, False

        else:
            raise Exception('mode should be train or test.')

    @staticmethod
    def get_coupon_history(_dir, user_file):

        dic = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        with open(_dir + user_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                _date = dt.datetime.strptime(row[1], '%Y%m%d')
                genre = row[2]
                week = int(row[3])
                count = int(row[4]) if row[4] != '' else None

                dic[_date][genre][week] = count

        return dic
