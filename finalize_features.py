# -*- coding: utf-8 -*-
"""
特征工程收尾: ① max_features 曲线见顶确认 ② 特征维度变化后 C 值重验
====================================================================
模型固定 LR, 划分/种子与全部前序实验一致 (6000/1368, stratify, seed=42)。
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE = 6000, 42

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

# ---------- ① 词表自然规模 & max_features 曲线收尾 ----------
vec_all = TfidfVectorizer(max_features=None, ngram_range=(1, 1))
vec_all.fit(X_tr)
print(f'训练集 6000 篇的 unigram 自然词表规模: {len(vec_all.vocabulary_)} 词')

print('\n=== max_features 曲线收尾 (ngram=(1,1)) ===')
for max_feat in [30000, 50000, None]:
    vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 1))
    Xtr = vec.fit_transform(X_tr)
    Xva = vec.transform(X_val)
    lr = LogisticRegression(C=100, max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
    acc = accuracy_score(y_val, lr.predict(Xva))
    print(f'max_features={str(max_feat):>6}: 词表 {len(vec.vocabulary_):>6} | 验证 {acc:.4f}')

# ---------- ② C 值重验 (在 30000 特征下) ----------
print('\n=== C 值重验 (max_features=30000) ===')
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 1))
Xtr = vec.fit_transform(X_tr)
Xva = vec.transform(X_val)
for C in [300, 1000, 3000]:
    lr = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
    acc = accuracy_score(y_val, lr.predict(Xva))
    print(f'C={C:>4}: 验证 {acc:.4f}')
