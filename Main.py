#! env python
# -*- coding: utf-8 -*-

import os
import sys
import configparser


from module import Predict as pred
from logging import getLogger, StreamHandler, DEBUG, FileHandler, Formatter

logger = getLogger('')
fileHandler = FileHandler(__file__.replace('py', 'log'))
fileHandler.setLevel(DEBUG)
formatter = Formatter('%(asctime)s - %(levelname)s - %(message)s')
fileHandler.setFormatter(formatter)
streamHandler = StreamHandler()
streamHandler.setLevel(DEBUG)
streamHandler.setFormatter(formatter)
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
logger.setLevel(DEBUG)

# ponpare.Main
# Date: 2018/03/26
# Filename: Main 

__author__ = 'takutohasegawa'
__date__ = "2018/03/26"


config = configparser.ConfigParser()
config.read_file(open('./config.conf', 'r', encoding='UTF-8'))


Predict = pred.Predict(config)


def output_model():
    # 予測モデルを作成し、予測結果を出力する
    Predict.output_model(param_search=True, sample=500, skip_cv=False, output_model=False)


def predict_output():

    # TODO: ビューティーが含まれない件
    model_file_path = './model/2018-04-13/2018-04-13-00-06-56'
    Predict.predict(model_file_path=model_file_path, sample=-1, output_submit_file=True,
                    separation=1000)


def predict_evaluate_output():
    # 予測モデルを作成し、提出用ファイルを出力する
    model_file_path = Predict.output_model(param_search=False, sample=10000, skip_cv=True,
                                           n_jobs=-1)

    # model_file_path = './model/2018-04-04/2018-04-04-00-18-48'
    Predict.predict(model_file_path=model_file_path, sample=-1, output_submit_file=True,
                    separation=1000)


def output_coef_file():

    import pickle
    model = pickle.load(open('./model/2018-04-13/2018-04-13-00-06-56', 'rb'))
    Predict.output_coef_file(model=model)


if __name__ == '__main__':
    output_model()
    # predict_output()
    # predict_evaluate_output()
    # output_coef_file()b

