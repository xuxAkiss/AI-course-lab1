# -*- coding: utf-8 -*-
"""
停用词实验: 怎么选词 + 效果测试
================================
1. 统计训练集里文档频率最高的词 (数据驱动的选词依据)
2. 对比 stop_words=None 与 stop_words='english' 的验证集准确率
配置与 tune_C.py 完全一致 (6000/1368, TF-IDF 5000, LR C=100)
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (CountVectorizer,
                                             ENGLISH_STOP_WORDS, TfidfVectorizer)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE, MAX_FEATURES = 6000, 42, 5000

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

# ---------- 第1部分: 数据驱动的停用词候选 ----------
cv = CountVectorizer(max_features=MAX_FEATURES, binary=True)  # binary: 只看"是否出现"
Xb = cv.fit_transform(X_tr)
doc_freq = np.asarray(Xb.sum(axis=0)).ravel()  # 每个词出现在多少篇文档里
names = cv.get_feature_names_out()
order = np.argsort(-doc_freq)

print(f'=== 6000 篇训练文档中出现面最广的 25 个词 ===')
for i in order[:25]:
    w = names[i]
    tag = ' [标准表内]' if w in ENGLISH_STOP_WORDS else ''
    print(f'{w:>12}: {doc_freq[i]:>4} 篇 ({doc_freq[i]/N_TRAIN:>6.1%}){tag}')

print('\n=== 你提议的三个词 ===')
for w in ['times', 'and', 'with']:
    idx = cv.vocabulary_.get(w)
    if idx is not None:
        tag = ' [标准表内]' if w in ENGLISH_STOP_WORDS else ' [不在标准表]'
        print(f'{w:>6}: {doc_freq[idx]:>4} 篇 ({doc_freq[idx]/N_TRAIN:>6.1%}){tag}')
    else:
        print(f'{w:>6}: 不在词表前5000')

# ---------- 第2部分: 效果对比 ----------
print('\n=== 停用词开关对比 (LR C=100) ===')
for sw in [None, 'english']:
    vec = TfidfVectorizer(max_features=MAX_FEATURES, stop_words=sw)
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_val)
    lr = LogisticRegression(C=100, max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
    acc = accuracy_score(y_val, lr.predict(Xva))
    print(f'stop_words={str(sw):>8}: 词表 {len(vec.vocabulary_):>5} | 验证准确率 {acc:.4f}')
