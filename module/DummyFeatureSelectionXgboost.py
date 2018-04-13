#! env python
# -*- coding: utf-8 -*-

from sklearn.base import BaseEstimator
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import copy
import pandas as pd
from logging import getLogger
from tqdm import tqdm

logger = getLogger('')

# ponpare.DummyFeatureSelectionLogisticRegression
# Date: 2018/03/26
# Filename: DummyLasso

__author__ = 'takutohasegawa'
__date__ = "2018/03/26"


class DummyFeatureSelectionXgboost(BaseEstimator):
    def __init__(self, separate_genre=True, max_depth=3, learning_rate=0.1, n_estimators=100, min_child_weight=1,
                 gamma=0, standardise=False, scale_pos_weight=0.1, dummy_drop_first=False, n_jobs=-1):

        self.separate_genre = separate_genre
        self.dummy_drop_first = dummy_drop_first
        self.standardise = standardise
        self.n_jobs = n_jobs

        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.scale_pos_weight = scale_pos_weight

        self.scaler = {}
        self.predictor = {}
        self.predictor_proba = {}
        self.colnames = defaultdict(list)
        self.importance = {}

        self.min_record_thr = 10

    def fit(self, X, y):
        logger.debug('START: fit')

        try:

            X.loc[:, 'y'] = y
            gp = X.groupby('GENRE_NAME') if self.separate_genre else [('all', X)]

            for genre, X in gp:
                logger.debug('START: {}'.format(genre))
                if len(X) < self.min_record_thr:
                    logger.debug('{}: DataFrame is too short. Skip.'.format(genre))
                    continue

                y = X['y']
                X = X.drop('y', axis=1)
                if self.separate_genre:
                    X = X.drop('GENRE_NAME', axis=1)

                logger.debug('START: get_dummies')
                X_dummy = pd.get_dummies(X, drop_first=self.dummy_drop_first, sparse=True)
                logger.debug('END: get_dummies')

                x_dummy_colnames = X_dummy.keys()

                X_dummy_not_scaled = None
                if self.standardise:
                    X_dummy_not_scaled = copy.deepcopy(X_dummy)
                    logger.debug('START: scale')
                    X_dummy = StandardScaler().fit_transform(X_dummy)
                    logger.debug('END: scale')
                    X_dummy = pd.DataFrame(X_dummy, columns=x_dummy_colnames)

                model = XGBClassifier(max_depth=self.max_depth, learning_rate=self.learning_rate,
                                      n_estimators=self.n_estimators, min_child_weight=self.min_child_weight,
                                      gamma=self.gamma, n_jobs=self.n_jobs, silent=False,
                                      scale_pos_weight=self.scale_pos_weight)

                logger.debug('START: model.fit *for feature selection')
                model.fit(X_dummy, y)
                logger.debug('END: model.fit *for feature selection')

                mask = []
                for i, (_imp, _col) in enumerate(zip(model.feature_importances_, x_dummy_colnames)):
                    if abs(_imp) >= model.feature_importances_.mean():
                        self.colnames[genre].append(_col)
                        mask.append(i)

                logger.debug('START: model.fit *after feature selection')
                model.fit(X_dummy[self.colnames[genre]], y)
                logger.debug('END: model.fit *after feature selection')

                if self.standardise:
                    logger.debug('START: scale *after feature selection')
                    self.scaler[genre] = StandardScaler().fit(X_dummy_not_scaled[self.colnames[genre]])
                    logger.debug('END: scale *after feature selection')
                self.predictor[genre] = model.predict
                self.predictor_proba[genre] = model.predict_proba
                self.importance[genre] = model.feature_importances_

        except:
            import traceback
            traceback.print_exc()
            pass

        return self

    def predict(self, X, yt=None):
        logger.debug('START: predict')

        # インデックスの付与
        X.loc[:, 'ix'] = list(range(len(X)))
        X.loc[:, 'y'] = yt

        # 分割
        logger.debug('START: group.by')
        if self.separate_genre:
            gp = X.groupby('GENRE_NAME')
        else:
            gp = [('all', X)]

        y_ix = []
        for genre, _X in gp:

            # インデックスを除く
            ix = _X['ix']

            if genre in self.colnames and genre in self.scaler:
                logger.debug('START: {}'.format(genre))

                # ダミー化
                logger.debug('START: get_dummies')
                X_dummy = pd.get_dummies(_X, drop_first=self.dummy_drop_first)
                X_dummy_cols = set(X_dummy.keys())

                # 足りない列を追加
                logger.debug('START: add_lacking_cols')
                for c in self.colnames[genre]:
                    if c not in X_dummy_cols:
                        X_dummy.loc[:, c] = [0 for _ in range(len(X_dummy))]

                # 標準化と不要なカラムの除去
                logger.debug('START: standardise')
                if self.standardise:
                    X_dummy = self.scaler[genre].transform(X_dummy[self.colnames[genre]])
                else:
                    X_dummy = X_dummy[self.colnames[genre]]

                logger.debug('START: predict')
                y = self.predictor[genre](X_dummy)

            else:  # 学習時に存在しないジャンルならば、全て0を返す
                y = [0 for _ in range(len(_X))]

            y_ix.extend([(_y, _ix) for _y, _ix in zip(y, ix)])

        y = [_y_ix[0] for _y_ix in sorted(y_ix, key=lambda x: x[1])]

        return y

    def predict_proba(self, X, yt=None):
        logger.debug('START: predict_proba')

        # インデックスの付与
        X.loc[:, 'ix'] = list(range(len(X)))
        X.loc[:, 'y'] = yt

        # 分割
        logger.debug('START: group.by')
        if self.separate_genre:
            gp = X.groupby('GENRE_NAME')
        else:
            gp = [('all', X)]

        y_ix = []
        for genre, _X in gp:

            # インデックスを除く
            ix = _X['ix']

            if genre in self.colnames:
                logger.debug('START: {}'.format(genre))

                # ダミー化
                logger.debug('START: get_dummies')
                X_dummy = pd.get_dummies(_X, drop_first=self.dummy_drop_first)
                X_dummy_cols = set(X_dummy.keys())

                # 足りない列を追加
                logger.debug('START: add_lacking_cols')
                for c in tqdm(self.colnames[genre]):
                    if c not in X_dummy_cols:
                        X_dummy.loc[:, c] = [0 for _ in range(len(X_dummy))]

                # 標準化と不要なカラムの除去
                if self.standardise:
                    logger.debug('START: standardise')
                    X_dummy = self.scaler[genre].transform(X_dummy[self.colnames[genre]])
                else:
                    X_dummy = X_dummy[self.colnames[genre]]

                logger.debug('START: predict_proba')
                y = [p[1] for p in self.predictor_proba[genre](X_dummy)]  # TODO: predict_probaは各クラスの確率を返す！

            else:  # 学習時に存在しないジャンルならば、全て0を返す
                y = [0 for _ in range(len(_X))]

            y_ix.extend([(_y, _ix) for _y, _ix in zip(y, ix)])

        y = [_y_ix[0] for _y_ix in sorted(y_ix, key=lambda x: x[1])]

        return y

