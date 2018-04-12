#! env python
# -*- coding: utf-8 -*-
# Filename: Preprocess

import csv, os, copy, random, configparser
from collections import defaultdict
from tqdm import tqdm
from module import Tools as tls
from module import MergeFileGenerator as mfg
import datetime as dt
import traceback
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

config = configparser.ConfigParser()
config.read_file(open('./config.conf', 'r', encoding='UTF-8'))
Tools = tls.Tools(config)


def output_coupon_purchase_history_file(_detail_file_path, debug=False):
    """クーポン購入履歴ファイルの出力"""

    cid_coupon_dic, label_dic = Tools.get_coupon_dic()

    # IDごとにクーポン購入履歴を取得
    dic = defaultdict(lambda: defaultdict(list))
    with open(_detail_file_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ix = {h: i for i, h in enumerate(header)}

        for i, row in tqdm(enumerate(reader)):
            if debug and i == 100:
                break
            uid = row[ix['USER_ID_hash']]
            _date = dt.datetime.strptime(row[ix['I_DATE']][:10], '%Y-%m-%d')
            cid = row[ix['COUPON_ID_hash']]

            dic[uid]['all'].append(_date)
            dic[uid][cid_coupon_dic[cid]['GENRE_NAME']].append(_date)

    # 週の開始からユーザーごとの購入履歴情報を集計
    purchase_count_dic = defaultdict(int)
    for uid, _dic in tqdm(dic.items()):
        for genre, purchase_dates in _dic.items():

            st_date = dt.datetime(2011, 7, 10)
            _date = st_date
            weeks = [1, 2, 4, 13, 26]
            while _date <= dt.datetime(2012, 6, 24):
                for week in weeks:
                    purchase_count_dic[(uid, _date.strftime('%Y%m%d'), genre, week)]\
                        = calc_count(purchase_dates, _date, week)
                _date += dt.timedelta(days=7)

    # 閲覧クーポン割合を付与
    # TODO:未実装（優先度低）

    # 出力
    fout = open(coupon_purchase_history_file_path, 'w', encoding='SJIS')
    writer = csv.writer(fout, delimiter=',', lineterminator='\n')
    writer.writerow(['uid', 'week_start', 'genre', 'prev_week_count', 'purchase'])
    for (uid, year_week, genre, week), cnt in tqdm(purchase_count_dic.items()):
        writer.writerow([uid, year_week, genre, week, cnt])

    fout.close()


def calc_count(purchase_dates, _date, week_count):

    cnt = 0
    st_date = _date - dt.timedelta(days=week_count*7)
    en_date = _date

    if st_date < dt.datetime(2011, 7, 3):
        return None

    else:
        for purchase_date in purchase_dates:
            if st_date <= purchase_date < en_date:
                cnt += 1
        return cnt


def output_coupon_visit_history_file(_visit_file_path, debug=False):

    # ID毎にクーポン閲覧履歴を取得
    cid_coupon_dic, label_dic = Tools.get_coupon_dic()

    # IDごとにクーポン閲覧履歴を取得
    dic = defaultdict(lambda: defaultdict(list))
    with open(_visit_file_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        ix = {h: i for i, h in enumerate(header)}

        for i, row in tqdm(enumerate(reader)):
            if debug and i == 100:
                break
            uid = row[ix['USER_ID_hash']]
            _date = dt.datetime.strptime(row[ix['I_DATE']][:10], '%Y-%m-%d')
            cid = row[ix['VIEW_COUPON_ID_hash']]
            sid = row[ix['SESSION_ID_hash']]

            dic[uid]['all'].append((_date, cid))
            if cid in cid_coupon_dic:
                dic[uid][cid_coupon_dic[cid]['GENRE_NAME']].append((_date, cid, sid))
            else:
                logger.debug('coupon id does not exist in coupon_dic: {}'.format(cid))

    # 週の開始からユーザーごとの購入履歴情報を集計
    visit_count_dic = defaultdict(int)
    for uid, _dic in tqdm(dic.items()):
        for genre, visit_date_cid_sids in _dic.items():

            st_date = dt.datetime(2011, 7, 10)
            _date = st_date
            weeks = [1, 2, 4, 13, 26]
            while _date <= dt.datetime(2012, 6, 24):
                for week in weeks:

                    # 閲覧クーポン-セッション数に変換
                    visit_dates, cid_sids = [], []
                    for tpl in visit_date_cid_sids:
                        if tpl[1:] not in cid_sids:
                            visit_dates.append(tpl[0]), cid_sids.append(tpl[1:])

                    visit_count_dic[(uid, _date.strftime('%Y%m%d'), genre, week)]\
                        = calc_count(visit_dates, _date, week)
                _date += dt.timedelta(days=7)

    # 閲覧クーポン割合を付与
    # TODO:未実装（優先度低）

    # 出力
    fout = open(coupon_visit_history_file_path, 'w', encoding='SJIS')
    writer = csv.writer(fout, delimiter=',', lineterminator='\n')
    writer.writerow(['uid', 'week_start', 'genre', 'prev_week_count', 'visit_coupon_session'])
    for (uid, year_week, genre, week), cnt in tqdm(visit_count_dic.items()):
        writer.writerow([uid, year_week, genre, week, cnt])

    fout.close()


def output_train_merge_files(debug=False, particular_files=None):
    """結合されたファイルの出力
    uid, start_date, クーポン購買履歴情報, クーポン閲覧履歴情報, cid, クーポン属性情報, 購入有無"""

    output_dir = './processed_data/merge_file_2/'
    os.makedirs(output_dir, exist_ok=True)

    MergeFileGenerator = mfg.MargeFileGenerator(config=config, mode='train', output_dir=output_dir)

    user_file_list = os.listdir(purchase_file_dir)
    existing_files = os.listdir(merge_file_dir)
    user_file_list = [user_file for user_file in user_file_list if user_file not in existing_files]
    # user_file_list = ['d56b6b3c994fef5395da2dbacf7bf17c.csv']

    if particular_files is not None:
        user_file_list = [user_file for user_file in particular_files]

    for i, user_file in tqdm(enumerate(user_file_list)):
        try:
            if debug and i == 10:
                break

            MergeFileGenerator.generate_file(user_file)

        except Exception as e:
            traceback.print_exc()
            logger.debug(e)


def output_test_merge_files(debug=False):

    particular_files = [s.replace(',\n', '.csv') for s in
                        open(config['GENERAL']['SAMPLE_SUBMISSION_FILE_PATH']).readlines()][1:]
    output_dir = './processed_data/test_merge_file/'
    os.makedirs(output_dir, exist_ok=True)

    existing_files = os.listdir(output_dir)
    particular_files = [user_file for user_file in particular_files if user_file not in existing_files]
    
    MergeFileGenerator = mfg.MargeFileGenerator(config=config, mode='test', output_dir=output_dir,
                                                date_from=dt.datetime(2012, 6, 24), date_until=dt.datetime(2012, 6, 24))

    # particular_files = ['000cc06982785a19e2a2fdb40b1c9d59.csv']
    for i, user_file in tqdm(enumerate(particular_files, start=1)):
        if debug and i == 100:
            break

        try:
            MergeFileGenerator.generate_file(user_file)

        except Exception as e:
            logger.debug(e)


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


def modify_header():

    filedir = './processed_data/merge_file/'
    outdir = './processed_data/merge_file_2/'
    files = os.listdir(filedir)
    for _file in tqdm(files):

        with open(filedir + _file, 'r') as f:

            reader = csv.reader(f)
            header = next(reader)
            # ix = {h: i for i, h in enumerate(header)}

            _header = header[:3] + ['SEX_ID', 'AGE', 'PREF_NAME'] + header[6:]

            fout = open(outdir + _file, 'w', encoding='SJIS')
            writer = csv.writer(fout, delimiter=',', lineterminator='\n')
            writer.writerow(_header)
            for row in reader:
                writer.writerow(row)

            fout.close()


if __name__ == '__main__':

    detail_file_path = './raw_data/coupon_detail_test.csv'
    coupon_purchase_history_file_path = './processed_data/coupon_purchase_history.csv'

    visit_file_path = './raw_data/coupon_visit_test.csv'
    coupon_visit_history_file_path = './processed_data/coupon_visit_history.csv'

    purchase_file_dir = './processed_data/coupon_purchase_history/'
    visit_file_dir = './processed_data/coupon_visit_history/'

    merge_file_dir = './processed_data/merge_file_2/'
    os.makedirs(merge_file_dir, exist_ok=True)

    # output_coupon_purchase_history_file(detail_file_path=detail_file_path, debug=False)
    # output_coupon_visit_history_file(visit_file_path=visit_file_path, debug=False)

    # output_merged_file(debug=False, paticular_files=None)
    # modify_header()

    # output_train_merge_files()
    output_test_merge_files(debug=False)
