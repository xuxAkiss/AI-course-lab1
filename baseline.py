# -*- coding: utf-8 -*-
"""
实验一 基线脚本：TF-IDF + 朴素贝叶斯
====================================
数据: train_cleaned.csv (已剥邮件头, 见 preprocess.py)
划分: 固定训练集 6000 条 / 验证集 1368 条, 分层抽样, random_state=42
流程: 仅用训练集 fit 向量化器 -> 训练模型 -> 验证集评估
"""
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 实验配置 (见 实验决策记录.md) ----------
N_TRAIN = 6000          # 固定训练集条数
RANDOM_STATE = 42       # 全局随机种子
MAX_FEATURES = 5000     # TF-IDF 词表上限 (与 draft_main.py 框架一致)
# --------------------------------------------------

# 1. 加载清洗后的数据
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

# 2. 划分验证集: 固定 6000 训练, 其余 1368 作验证, 按类别分层
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=N_TRAIN, stratify=y, random_state=RANDOM_STATE)
print(f'训练集 {len(X_tr)} 条 | 验证集 {len(X_val)} 条')

# 3. TF-IDF: 只在训练集上 fit (防止数据泄露)
vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)
X_tr_tfidf = vectorizer.fit_transform(X_tr)
X_val_tfidf = vectorizer.transform(X_val)
print(f'TF-IDF 词表大小: {len(vectorizer.vocabulary_)}')

# 4. 基线模型: 朴素贝叶斯 (默认参数)
model = MultinomialNB()
model.fit(X_tr_tfidf, y_tr)
y_val_pred = model.predict(X_val_tfidf)

# 5. 验证集评估
acc = accuracy_score(y_val, y_val_pred)
print(f'\n验证集准确率: {acc:.4f}')
print(classification_report(y_val, y_val_pred, digits=4))
