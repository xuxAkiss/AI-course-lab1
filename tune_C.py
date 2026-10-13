# -*- coding: utf-8 -*-
"""
调参演示: 在固定验证集上扫 C 值 (SVM 线性核 vs 逻辑回归)
======================================================
划分与 baseline.py 完全一致 (6000/1368, stratify, seed=42),
只改变 C 这一个变量, 同时报告训练集与验证集准确率用于诊断过/欠拟合。
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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

C_LIST = [0.01, 0.1, 1, 10, 100, 300, 1000]
print('基线(朴素贝叶斯)验证集准确率: 0.8984')
print()
print(f"{'C':>7} | {'SVM训练':>8} {'SVM验证':>8} | {'LR训练':>8} {'LR验证':>8}")
print('-' * 52)
for C in C_LIST:
    svm = SVC(kernel='linear', C=C, random_state=RANDOM_STATE).fit(X_tr_tfidf, y_tr)
    lr = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE).fit(X_tr_tfidf, y_tr)
    print(f"{C:>7.4g} | "
          f"{svm.score(X_tr_tfidf, y_tr):>8.4f} "
          f"{accuracy_score(y_val, svm.predict(X_val_tfidf)):>8.4f} | "
          f"{lr.score(X_tr_tfidf, y_tr):>8.4f} "
          f"{accuracy_score(y_val, lr.predict(X_val_tfidf)):>8.4f}")
