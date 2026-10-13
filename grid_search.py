# -*- coding: utf-8 -*-
"""
特征工程网格扫参: max_features × ngram_range (模型固定 LR C=100)
==============================================================
划分/种子与之前实验完全一致 (6000/1368, stratify, seed=42)。
8 组配置, 结果实时打印到 grid_search.log。
"""
import itertools
import os
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE = 6000, 42
LOG = os.path.join(BASE_DIR, 'grid_search.log')

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

# 第2轮扩展: bigram 已证伪(见决策12), 向右找 max_features 峰值
GRID = [(15000, (1, 1)), (20000, (1, 1)), (30000, (1, 1)), (30000, (1, 2))]

with open(LOG, 'w', encoding='utf-8') as log:
    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say('特征工程网格扫参 (LR C=100), 基线: max_features=5000, ngram=(1,1) -> 0.9189')
    say('')
    results = []
    for max_feat, ngram in GRID:
        t0 = time.time()
        vec = TfidfVectorizer(max_features=max_feat, ngram_range=ngram)
        Xtr = vec.fit_transform(X_tr)
        Xva = vec.transform(X_val)
        lr = LogisticRegression(C=100, max_iter=2000, random_state=RANDOM_STATE)
        lr.fit(Xtr, y_tr)
        acc = accuracy_score(y_val, lr.predict(Xva))
        dt = time.time() - t0
        results.append((max_feat, ngram, acc, dt))
        say(f'max_features={max_feat:>6} ngram={ngram} | 验证集 {acc:.4f} | 用时 {dt:>4.0f}s')

    say('')
    say('=== 按准确率排序 ===')
    for max_feat, ngram, acc, dt in sorted(results, key=lambda r: -r[2]):
        say(f'  {acc:.4f}  max_features={max_feat:>6} ngram={ngram}')
    say('')
    say('全部完成')
