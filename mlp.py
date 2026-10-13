# -*- coding: utf-8 -*-
"""
MLP 补充实验 (思考题2): 网络结构 / 训练轮数对性能的影响
======================================================
特征固定为 TF-IDF(30000, unigram), 划分/种子与全实验一致 (6000/1368, seed=42)。
对照: LR C=300 验证集 0.9393。
每组报告训练集/验证集准确率, 用于诊断过拟合。
"""
import os
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE, MAX_FEATURES = 6000, 42, 30000
LOG = os.path.join(BASE_DIR, 'mlp.log')

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 1))
Xtr = vec.fit_transform(X_tr)
Xva = vec.transform(X_val)

# (hidden_layer_sizes, max_iter) 组合: 默认 -> 加宽 -> 加深 -> 加长训练
CONFIGS = [
    ('默认 (100,) iter=300', dict(hidden_layer_sizes=(100,), max_iter=300)),
    ('加宽 (200,) iter=300', dict(hidden_layer_sizes=(200,), max_iter=300)),
    ('加深 (100,50) iter=300', dict(hidden_layer_sizes=(100, 50), max_iter=300)),
    ('加长 (100,) iter=1000', dict(hidden_layer_sizes=(100,), max_iter=1000)),
]

with open(LOG, 'w', encoding='utf-8') as log:
    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say('MLP 补充实验, 对照 LR C=300: 验证集 0.9393')
    say('')
    for name, kwargs in CONFIGS:
        t0 = time.time()
        mlp = MLPClassifier(random_state=RANDOM_STATE, **kwargs)
        mlp.fit(Xtr, y_tr)
        tr_acc = accuracy_score(y_tr, mlp.predict(Xtr))
        va_acc = accuracy_score(y_val, mlp.predict(Xva))
        dt = time.time() - t0
        say(f'{name:>24}: 训练 {tr_acc:.4f} | 验证 {va_acc:.4f} | 用时 {dt:>5.0f}s')
    say('')
    say('全部完成')
