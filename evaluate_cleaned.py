# -*- coding: utf-8 -*-
"""
清洗 v2 效果评估: 噪声词残留统计 + LR C=100 验证集对比
====================================================
对比旧清洗(只有剥邮件头) vs 新清洗(剥头部+Subject标签+Re:前缀+引用归属行)
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE, MAX_FEATURES = 6000, 42, 5000

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

# 数据完整性检查
empty = sum(len(t.strip()) == 0 for t in X)
print(f'完整性: 行数 {len(X)}, 空文本 {empty} 条, 平均长度 {np.mean([len(t) for t in X]):.0f} 字符')

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

# 噪声词残留统计
cv = CountVectorizer(max_features=MAX_FEATURES, binary=True)
Xb = cv.fit_transform(X_tr)
doc_freq = np.asarray(Xb.sum(axis=0)).ravel()
print('\n噪声词在新清洗数据中的文档频率:')
for w in ['subject', 're', 'writes', 'wrote', 'edu']:
    idx = cv.vocabulary_.get(w)
    if idx is not None:
        print(f'  {w:>8}: {doc_freq[idx]:>4} 篇 ({doc_freq[idx]/N_TRAIN:.1%})')
    else:
        print(f'  {w:>8}: 已不在前5000词表')

# LR C=100 对比
vec = TfidfVectorizer(max_features=MAX_FEATURES)
Xtr = vec.fit_transform(X_tr)
Xva = vec.transform(X_val)
lr = LogisticRegression(C=100, max_iter=2000, random_state=RANDOM_STATE).fit(Xtr, y_tr)
acc = accuracy_score(y_val, lr.predict(Xva))
print(f'\nLR C=100 验证集准确率 (新清洗数据): {acc:.4f}')
print(f'旧清洗数据上的成绩: 0.9189')
