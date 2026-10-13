# -*- coding: utf-8 -*-
"""
查看模型学到了什么: 打印每个类别权重最高的词
============================================
与 tune_C.py 相同配置 (6000/1368, TF-IDF 5000, LR C=100),
训练完成后, coef_[类别][词] 就是"这个词为该类别投的票"的权重。
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, RANDOM_STATE, MAX_FEATURES = 6000, 42, 5000

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)

vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_val_tfidf = vectorizer.transform(X_val)

lr = LogisticRegression(C=100, max_iter=2000, random_state=RANDOM_STATE)
lr.fit(X_tr_tfidf, y_tr)

coef = lr.coef_           # 形状 (10, 5000): 每行是一个类别的"每个词的投票权重"
words = np.array(vectorizer.get_feature_names_out())

print('逻辑回归 C=100 训练完成。每个类别权重最高的 8 个词:\n')
for c in range(10):
    top_idx = np.argsort(coef[c])[-8:][::-1]
    top_words = ', '.join(words[top_idx])
    print(f'类别 {c}: {top_words}')

# 演示: 一条新文本怎么打分
print('\n演示: 对文本 "my car has a new engine" 的各类别得分(前3名):')
demo_vec = vectorizer.transform(['my car has a new engine'])
scores = lr.decision_function(demo_vec)[0]
top3 = np.argsort(scores)[-3:][::-1]
for c in top3:
    print(f'  类别 {c}: 得分 {scores[c]:.2f}')
print(f'  -> 模型预测: {top3[0]}')
