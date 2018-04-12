#! env python
# -*- coding: utf-8 -*-

import os, random, csv, pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from collections import defaultdict
from logging import getLogger, StreamHandler, DEBUG, FileHandler, Formatter
from module import DummyFeatureSelectionLogisticRegression
from module import DummyFeatureSelectionXgboost

# ponpare.Predict
# Date: 2018/03/26
# Filename: Predict 

__author__ = 'takutohasegawa'
__date__ = "2018/03/26"


class Predict:
    def __init__(self, config):

        self.config = config
        self.logger = getLogger(__name__)
        self.trial_time = dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.model_type = config['PREDICT']['MODEL_TYPE']

    def predict(self, model_file_path, sample=-1, output_submit_file=False, separation=-1,
                debug=False):

        # データ読み込み
        files_list = []
        merge_file_dir = self.config['GENERAL']['MERGE_FILE_DIR_TEST']
        all_files = os.listdir(merge_file_dir)

        if separation < 0:
            files_list.append(all_files)
        else:
            start = 0
            while start < len(all_files):
                files_list.append(all_files[start:min(start+separation, len(all_files)-1)])
                start += separation

        model = pickle.load(open(model_file_path, 'rb'))

        y_pred, uids, cids = [], [], []
        for i, files in enumerate(files_list, start=1):
            # if i == 3:
            #     break

            X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test = \
                self.get_data(merge_file_dir=merge_file_dir, sample=sample, train_ratio=1,
                              particular_files=files)

            _y_pred = model.predict_proba(X_train)
            y_pred.extend(_y_pred), uids.extend(uids_train), cids.extend(cids_train)

        if output_submit_file:
            self.output_submit_file(y_pred, uids, cids, format_submit_file=True)

    def output_model(self, param_search=False, sample=-1, skip_cv=False, n_jobs=-1):
        """モデルファイルを出力する"""

        if param_search:
            param_grid = [{'penalty': ['l1'],
                           'C': [0.5],
                           'class_weight': [None],
                           'standardise': [True],
                           'separate_genre': [True],
                           'dummy_drop_first': [False],
                           'min_coef_value': [0.001],
                           'solver': ['saga']}]  # TODO:sagaにするとn_jobs=-1にできる

            # param_grid = [{'max_depth': [3],
            #                'learning_rate': [0.1],
            #                'n_estimators': [100],
            #                'gamma': [0],
            #                'separate_genre': [False, True]}]

            params = self.param_search(param_grid=param_grid, sample=sample, output_evaluation_file=True,
                                       skip_cv=skip_cv, n_jobs=n_jobs)

        else:
            params = {'penalty': 'l1', 'C': 1, 'class_weight': None, 'standardise': True,
                            'separate_genre': True, 'min_coef_value': 0.001}
            self.logger.debug(params)

        return self.learn_model(params=params, sample=sample)

    def param_search(self, param_grid, sample=-1, output_evaluation_file=False, skip_cv=False, n_jobs=-1):
        """学習モデルを作成し、評価結果(MAP@10)を返す"""

        # 出力先ディレクトリ生成
        today_dir = self.config['GENERAL']['MODEL_FILE_DIR'] + dt.datetime.today().strftime('%Y-%m-%d') + '/'
        os.makedirs(today_dir, exist_ok=True)

        # データ読み込み
        X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test = \
            self.get_data(merge_file_dir=self.config['GENERAL']['MERGE_FILE_DIR_TRAIN'],
                           sample=sample, train_ratio=0.9)

        # モデル学習
        self.logger.debug('sample: {0}\nparam_grid:{1}'.format(sample, param_grid))

        # スコアの作成
        # my_score = make_scorer(self.my_score, greater_is_better=True)
        if not skip_cv:

            if self.model_type == 'lr':
                grid = GridSearchCV(DummyFeatureSelectionLogisticRegression.
                                    DummyFeatureSelectionLogisticRegression(n_jobs=n_jobs), param_grid, cv=5,
                                    scoring='neg_log_loss', verbose=10, n_jobs=n_jobs)

            elif self.model_type == 'xb':
                grid = GridSearchCV(DummyFeatureSelectionXgboost.DummyFeatureSelectionXgboost(n_jobs=n_jobs),
                                    param_grid=param_grid, cv=5, scoring='neg_log_loss', verbose=10, n_jobs=n_jobs)

            else:
                raise ValueError

            grid.fit(X_train, y_train)
            self.logger.debug(grid.cv_results_)
            model = grid.best_estimator_
            print(grid.best_params_)
            print(grid.best_score_)
            params = grid.best_params_

            # 出力
            open(today_dir + 'grid_result_{}.txt'.format(self.trial_time), 'w') \
                .writelines([str(t) + '\n' for t in grid.cv_results_.items()])

            if self.model_type == 'lr':
                self.output_coef_file(model)
            elif self.model_type == 'xb':
                self.output_importance_file(model)

        else:
            params = {param: l[0] for param, l in param_grid[0].items()}
            self.logger.debug(params)

            if self.model_type == 'lr':
                model = DummyFeatureSelectionLogisticRegression.DummyFeatureSelectionLogisticRegression(**params).\
                    fit(X_train, y_train)

            elif self.model_type == 'xb':
                model = DummyFeatureSelectionXgboost.DummyFeatureSelectionXgboost(**params).\
                    fit(X_train, y_train)
            else:
                raise ValueError

        y_train_pred = model.predict_proba(X_train)
        y_test_pred = model.predict_proba(X_test)

        # submit形式に変換し、MAP@10を算出
        pred_uid_cid_dic_train = self.format_submit_file(y_train_pred, uids_train, cids_train, pred_thr=0)
        pred_uid_cid_dic_test = self.format_submit_file(y_test_pred, uids_test, cids_test, pred_thr=0)

        Map10_train = self.calc_map10(pred_uid_cid_dic_train, uids_train, cids_train, y_train)
        Map10_test = self.calc_map10(pred_uid_cid_dic_test, uids_test, cids_test, y_test)
        print('MAP@10 train: {0}\nMAP@10 test: {1}'.format(Map10_train, Map10_test))

        if output_evaluation_file:
            self.output_evaluation_file(X=X_test, y_pred=y_test_pred, y_true=y_test, uids=uids_test, cids=cids_test)

        return params

    def learn_model(self, params, sample=-1):

        # 出力先ディレクトリ生成
        today_dir = self.config['GENERAL']['MODEL_FILE_DIR'] + dt.datetime.today().strftime('%Y-%m-%d') + '/'
        os.makedirs(today_dir, exist_ok=True)

        X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test = \
            self.get_data(merge_file_dir=self.config['GENERAL']['MERGE_FILE_DIR_TRAIN'],
                          sample=sample, train_ratio=1.)

        if self.model_type == 'lr':
            model = DummyFeatureSelectionLogisticRegression.DummyFeatureSelectionLogisticRegression(**params). \
                fit(X_train, y_train)
            self.output_coef_file(model)

        elif self.model_type == 'xb':
            model = DummyFeatureSelectionXgboost.DummyFeatureSelectionXgboost(**params). \
                fit(X_train, y_train)
            self.output_importance_file(model)

        else:
            raise ValueError

        pickle.dump(model, open(today_dir + self.trial_time, 'wb'))

        return today_dir + self.trial_time

    def get_data(self, merge_file_dir, sample=-1, train_ratio=0.8, seed=0, particular_files=None):
        """前処理後データディレクトリからデータの読み込み、指定した人数のデータを読み込む
            指定した比率で学習データ、テストデータに分割する"""

        self.logger.debug('reading data...')
        if particular_files is not None:
            files = particular_files
        else:
            files = os.listdir(merge_file_dir)
            if sample > 0:
                random.seed(seed)
                files = random.sample(files, sample)

        row_list, y, colname, uids, cids = [], [], None, [], []
        initial_var = self.config['PREDICT']['INITIAL_VAR'].replace(' ', '').split(',')
        continuous_var = self.config['PREDICT']['CONTINUOUS_VAR'].replace(' ', '').split(',')

        for i, _file in tqdm(enumerate(files)):

            try:
                with open(merge_file_dir + _file, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)
                    ix = {h: i for i, h in enumerate(header)}

                    for row in reader:
                        _row = []
                        if '' in row[ix['purchase_all_1']:ix['visit_健康・医療_26']+1]:
                            # もし、purchaseかvisitが''なら、そのレコードをとばす
                            continue

                        for var in initial_var:
                            if var in continuous_var:
                                v = float(row[ix[var]]) if (row[ix[var]] != '' and row[ix[var]] != 'NA') else None
                            else:
                                v = row[ix[var]] if (row[ix[var]] != '' and row[ix[var]] != 'NA') else None

                            _row.append(v)

                        row_list.append(_row)
                        y.append(int(row[ix['purchase']])), uids.append(row[ix['uid']]), cids.append(row[ix['cid']])

            except Exception as e:
                self.logger.debug(e)

        X = pd.DataFrame(row_list, columns=initial_var)

        # 欠損値処理
        self.logger.debug('START: fillna')
        X = X.fillna({'VALIDPERIOD': 365, 'USABLE_DATE_MON': '1', 'USABLE_DATE_TUE': '1', 'USABLE_DATE_WED': '1',
                      'USABLE_DATE_THU': '1', 'USABLE_DATE_FRI': '1', 'USABLE_DATE_SAT': '1', 'USABLE_DATE_SUN': '1',
                      'USABLE_DATE_HOLIDAY': '1', 'USABLE_DATE_BEFORE_HOLIDAY': '1'})

        if self.config['PREDICT']['ADD_SAME_GENRE'] != '':
            self.logger.debug('START: add_same_genre_purchase_visit')
            X = self.add_same_genre_purchase_visit(X)

        if self.config['PREDICT']['ADD_PLACE_INTERACTION'] == 'TRUE':
            self.logger.debug('START: add_place_interatction')
            X = self.add_place_interaction(X)

        if train_ratio < 1:
            X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test \
                = train_test_split(X, y, uids, cids, train_size=train_ratio, random_state=seed)
        else:
            X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test \
                = X, None, y, None, uids, None, cids, None

        return X_train, X_test, y_train, y_test, uids_train, uids_test, cids_train, cids_test

    @staticmethod
    def format_submit_file(y_pred, uid, cid, upper=10, pred_thr=0.25):
        """
        購入確率の予測値を取得し、その場で上位10件and/or閾値以上のクーポンを{uid: [cid, cid,...]}の形式で出力する
        """

        dic = defaultdict(list)
        for _uid, _cid, _y in zip(uid, cid, y_pred):
            dic[_uid].append((_cid, _y))

        dic2 = defaultdict(list)
        for _uid, cid_y in dic.items():
            for i, (_cid, _y) in enumerate(sorted(cid_y, key=lambda x: x[1], reverse=True), start=1):
                if (i > upper) or (_y < pred_thr):
                    break
                dic2[_uid].append(_cid)

        return dic2

    def output_submit_file(self, y_pred, uids, cids, format_submit_file=True):
        """提出用ファイルを出力する"""

        dic = self.format_submit_file(y_pred, uids, cids)
        # dic = defaultdict(list)
        # for _uid, _cid, _y in zip(uid, cid, y_pred):
        #     if _y == 1:
        #         dic[_uid].append(_cid)

        fout = open(self.config['GENERAL']['SUBMIT_FILE_DIR'] + 'submission_{}.csv'
                    .format(dt.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')), 'w', encoding='UTF-8')
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(['USER_ID_hash', 'PURCHASED_COUPONS'])

        uid_list = [s.replace(',\n', '') for s in
                            open(self.config['GENERAL']['SAMPLE_SUBMISSION_FILE_PATH']).readlines()][1:] \
                            if format_submit_file else set(uids)

        for _uid in uid_list:
            writer.writerow([_uid, ' '.join(dic[_uid])])

        fout.close()

    def output_coef_file(self, model):
        """回帰式の係数ファイルを出力する"""
        # 出力先ディレクトリ生成
        today_dir = self.config['GENERAL']['MODEL_FILE_DIR'] + dt.datetime.today().strftime('%Y-%m-%d') + '/'
        os.makedirs(today_dir, exist_ok=True)

        fout = open(today_dir + 'coef_{}.csv'.
                    format(self.trial_time), 'w', encoding='SJIS')
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(['model', 'coef', 'value'])
        for genre in model.colnames.keys():
            for col, _coef in zip(model.colnames[genre], model.coef[genre]):
                writer.writerow([genre, col, _coef])
        fout.close()

    def output_importance_file(self, model):
        """回帰式の係数ファイルを出力する"""
        # 出力先ディレクトリ生成
        today_dir = self.config['GENERAL']['MODEL_FILE_DIR'] + dt.datetime.today().strftime('%Y-%m-%d') + '/'
        os.makedirs(today_dir, exist_ok=True)

        fout = open(today_dir + 'importance_{}.csv'.
                    format(self.trial_time), 'w', encoding='SJIS')
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(['model', 'coef', 'value'])
        for genre in model.colnames.keys():
            for col, imp in zip(model.colnames[genre], model.importance[genre]):
                writer.writerow([genre, col, imp])
        fout.close()

    def output_evaluation_file(self, X, y_pred, y_true, uids, cids):
        """評価用ファイルを出力する"""

        X.loc[:, 'y_pred'] = y_pred
        X.loc[:, 'y'] = y_true
        X.loc[:, 'uid'] = uids
        X.loc[:, 'cid'] = cids

        # 出力先ディレクトリ生成
        today_dir = self.config['GENERAL']['MODEL_FILE_DIR'] + dt.datetime.today().strftime('%Y-%m-%d') + '/'
        os.makedirs(today_dir, exist_ok=True)

        fout = open(today_dir + 'evaluation_' + self.trial_time + '.csv', 'w', encoding='SJIS')
        writer = csv.writer(fout, delimiter=',', lineterminator='\n')
        writer.writerow(X.keys())
        for row in X.values:
            writer.writerow(row)
        fout.close()

    @staticmethod
    def calc_map10(pred_uid_cid_dic, uids, cids, y):
        """MAP@10を算出する"""

        dic = defaultdict(lambda: {'pred': [], 'true': []})
        for uid, cid, _y in zip(uids, cids, y):
            if _y == 1:
                dic[uid]['true'].append(cid)

        for uid, cids in pred_uid_cid_dic.items():
            dic[uid]['pred'] = pred_uid_cid_dic[uid]

        MAP10 = 0
        for uid, _dic in dic.items():
            P = 0
            for i, cid_pred in enumerate(_dic['pred'], start=1):
                if i == 10:
                    break
                if cid_pred in _dic['true']:
                    P += 1

            MAP10 += P/min(10, len(_dic['true'])) if len(_dic['true']) > 0 else 0

        MAP10 = MAP10/len(dic)

        return MAP10

    @staticmethod
    def my_score(y, y_pred):
        """yが1の場合の場合のみの正答率を算出する"""

        _score = 0
        for _y, _y_pred in zip(y, y_pred):
            if _y == 1:
                if _y_pred == 1:
                    _score += 1
        return _score/sum(y)

    def add_same_genre_purchase_visit(self, X):
        """対象クーポンと同じジャンルのクーポンを購入または閲覧した数の付与"""

        ix = {c: i for i, c in enumerate(X.keys())}
        add_same_genre = self.config['PREDICT']['ADD_SAME_GENRE'].replace(' ', '').split(',')

        row_list = []
        for row in X.values:
            _row = []
            genre = row[ix['GENRE_NAME']]
            for same_genre in add_same_genre:
                _row.append(row[ix[same_genre.replace('same', genre)]])
            row_list.append(_row)

        X_add = pd.DataFrame(row_list, columns=add_same_genre)
        X = pd.concat([X, X_add], axis=1)

        return X

    @staticmethod
    def add_place_interaction(X):
        """ユーザーとクーポンの場所の交互作用項を加える"""

        PREF_NAME_large_area_name, PREF_NAME_ken_name, PREF_NAME_small_area_name = [], [], []
        for PREF_NAME, large_area_name, ken_name, small_area_name in \
                zip(X['PREF_NAME'], X['large_area_name'], X['ken_name'], X['small_area_name']):

            PREF_NAME_large_area_name.append(str(PREF_NAME) + '-' + str(large_area_name))
            PREF_NAME_ken_name.append(str(PREF_NAME) + '-' + str(ken_name))
            PREF_NAME_small_area_name.append(str(PREF_NAME) + '-' + str(small_area_name))

        X.loc[:, 'PREF_NAME_large_area_name'] = PREF_NAME_large_area_name
        X.loc[:, 'PREF_NAME_ken_name'] = PREF_NAME_ken_name
        X.loc[:, 'PREF_NAME_small_area_name'] = PREF_NAME_small_area_name

        return X







