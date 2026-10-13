# -*- coding: utf-8 -*-
"""
实验一 主脚本: 最终配置全量重训 + 生成提交文件 predictions.csv
============================================================
锁定配置 (决策18, 详见 实验决策记录.md):
    v1 清洗 (preprocess.py) + TF-IDF(max_features=30000, ngram=(1,1))
    + MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
验证集表现 (6000/1368 划分): 单划分 94.01% (seed=42);
    5 种子复验均值 94.28% ± 0.29%, 全部种子优于 LR(C=300) 的 93.73% ± 0.55% (决策17)
备选: MODEL='lr' 可切回 LogisticRegression(C=300) (93.93%)
复现: python main.py  (自动从原始 CSV 清洗, 无需其他前置步骤)
"""
import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE = 42       # 全局随机种子 (评分要求固定种子)
MAX_FEATURES = 30000    # TF-IDF 词表上限 (决策13)
C = 300                 # 逻辑回归正则化倒数 (决策13)
MODEL = 'mlp'           # 最终提交模型: 'mlp' (决策18) 或 'lr' (决策13备选)

np.random.seed(RANDOM_STATE)


def main():
    # 1. 加载原始数据, 应用 v1 清洗 (决策1~11)
    train_df, test_df = preprocess.load_raw()
    X_train = train_df['text'].astype(str).map(preprocess.clean_text).tolist()
    y_train = train_df['target'].values
    X_test = test_df['text'].astype(str).map(preprocess.clean_text).tolist()
    print(f'训练集 {len(X_train)} 条 (全量) | 测试集 {len(X_test)} 条')

    # 2. TF-IDF 向量化 (只在训练集上 fit, 防数据泄露)
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 1))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f'TF-IDF 词表 {len(vectorizer.vocabulary_)} 词')

    # 3. 最终模型: 全量训练集重训
    #    (验证集仅用于模型选择与调参, 最终模型用全部标注数据训练)
    if MODEL == 'lr':
        model = LogisticRegression(C=C, max_iter=2000, random_state=RANDOM_STATE)
    else:
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,
                              random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)
    print(f'训练集准确率(自检): {model.score(X_train_tfidf, y_train):.4f}')

    # 4. 测试集预测并保存 (格式与示例 predictions.csv 一致: 无表头, 单列)
    predictions = model.predict(X_test_tfidf)
    out_path = os.path.join(BASE_DIR, 'predictions.csv')
    pd.DataFrame(predictions).to_csv(out_path, index=False, header=False)
    print(f'已保存 {out_path}, 共 {len(predictions)} 条')
    print('预测类别分布:', pd.Series(predictions).value_counts().sort_index().to_dict())


if __name__ == '__main__':
    main()
