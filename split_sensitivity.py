# -*- coding: utf-8 -*-
"""
划分敏感性复验 (不确定性与局限性分析)
====================================
对最终 LR(C=300) 与 MLP(100,) 用 5 个随机种子重复 6000/1368 划分,
报告验证集准确率的均值 ± 标准差。
注: MLP(200,) 单种子训练需 95 分钟, 5 种子代价过大, 用 MLP(100,) 作代表;
    (200,) 的补充种子复验见后续。
"""
import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
N_TRAIN, MAX_FEATURES = 6000, 30000
SEEDS = [42, 2024, 7, 123, 999]
LOG = os.path.join(BASE_DIR, 'split_sensitivity.log')

train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values

# 断点续跑: 读取已有日志中完成的种子, 跳过它们
done_seeds = set()
lr_accs, mlp_accs = [], []
if os.path.exists(LOG):
    with open(LOG, encoding='utf-8') as f:
        for line in f:
            parts = line.split('|')
            if len(parts) == 4 and parts[0].strip().isdigit():
                seed = int(parts[0].strip())
                done_seeds.add(seed)
                lr_accs.append(float(parts[1].strip()))
                mlp_accs.append(float(parts[2].strip()))
if done_seeds:
    print(f'检测到已完成种子: {sorted(done_seeds)}, 跳过续跑')

with open(LOG, 'a', encoding='utf-8') as log:
    def say(msg):
        print(msg, flush=True)
        log.write(msg + '\n')
        log.flush()

    say('划分敏感性复验: LR C=300 与 MLP(100,) 各 5 个种子')
    say(f'{"seed":>6} | {"LR val":>8} | {"MLP(100,) val":>12} | 用时')
    say('-' * 50)
    for seed in SEEDS:
        if seed in done_seeds:
            continue
        t0 = time.time()
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, train_size=N_TRAIN, stratify=y, random_state=seed)
        vec = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=(1, 1))
        Xtr = vec.fit_transform(X_tr)
        Xva = vec.transform(X_val)

        lr = LogisticRegression(C=300, max_iter=2000, random_state=42)
        lr.fit(Xtr, y_tr)
        lr_acc = accuracy_score(y_val, lr.predict(Xva))
        lr_accs.append(lr_acc)

        mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=42)
        mlp.fit(Xtr, y_tr)
        mlp_acc = accuracy_score(y_val, mlp.predict(Xva))
        mlp_accs.append(mlp_acc)

        say(f'{seed:>6} | {lr_acc:>8.4f} | {mlp_acc:>12.4f} | {time.time()-t0:>5.0f}s')

    say('-' * 50)
    say(f'LR     均值 {np.mean(lr_accs):.4f} ± {np.std(lr_accs):.4f}')
    say(f'MLP(100) 均值 {np.mean(mlp_accs):.4f} ± {np.std(mlp_accs):.4f}')
    say('全部完成')
