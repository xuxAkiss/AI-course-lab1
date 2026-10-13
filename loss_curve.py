# -*- coding: utf-8 -*-
"""
最终模型训练损失曲线 (老师要求的"每一步损失函数变化图")
====================================================
配置与 main.py 完全一致: 全量 7368 条, TF-IDF 30000, MLP(100,) it=300, seed=42。
loss_curve_ 是 sklearn 训练过程中逐轮(epoch)真实记录的损失, 原样绘图不做平滑。
原始数据同时保存到 loss_curve.csv 便于核对。
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RANDOM_STATE, MAX_FEATURES = 42, 30000

np.random.seed(RANDOM_STATE)

# 与 main.py 一致: 原始 CSV + v1 清洗
CSV_PATH = os.path.join(BASE_DIR, 'loss_curve.csv')
if os.path.exists(CSV_PATH):   # 已有原始数据则只重新绘图, 不重复训练
    loss = pd.read_csv(CSV_PATH)['loss'].values
    print(f'复用已保存的 {len(loss)} 轮损失数据')
else:
    train_df, _ = preprocess.load_raw()
    X_train = train_df['text'].astype(str).map(preprocess.clean_text).tolist()
    y_train = train_df['target'].values

    vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 1))
    Xtr = vec.fit_transform(X_train)

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,
                        random_state=RANDOM_STATE)
    mlp.fit(Xtr, y_train)

    loss = mlp.loss_curve_
    pd.DataFrame({'epoch': np.arange(1, len(loss) + 1), 'loss': loss}).to_csv(
        CSV_PATH, index=False)

print(f'训练完成: {len(loss)} 轮, 初始损失 {loss[0]:.4f}, 最终损失 {loss[-1]:.4f}')

fig, ax = plt.subplots(figsize=(7, 3.4))
ax.plot(range(1, len(loss) + 1), loss, lw=0.8, color='#c0392b')
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title(f'Final Model (MLP 100,) Training Loss - '
             f'early-stopped at epoch {len(loss)} (max_iter=300), full 7368 samples')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'figures', 'fig_loss_curve.png'))
print('已保存 figures/fig_loss_curve.png 与 loss_curve.csv')
