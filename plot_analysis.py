# -*- coding: utf-8 -*-
"""
生成实验报告可视化图表 (评分项: 训练结果 / 参数对比 / 分类性能分析)
================================================================
输出到 figures/ 目录, 共 7 张 PNG。
所有数据来自本实验的真实记录 (实验决策记录.md), 非示意数据。
"""
import os

import matplotlib
matplotlib.use('Agg')  # 无窗口环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({'figure.dpi': 150, 'axes.grid': True,
                     'grid.alpha': 0.3, 'font.size': 9})

# ================= 图5: 实验进化链 (训练结果) =================
stages = ['NB\nbaseline', 'LR C=100\n5000 feat', 'LR C=100\n30000 feat',
          'LR C=300\n30000 feat', 'MLP(100,)\n30000 feat']
accs = [0.8984, 0.9189, 0.9349, 0.9393, 0.9401]
fig, ax = plt.subplots(figsize=(7.5, 3.8))
bars = ax.bar(stages, accs, color=['#8fa8c8', '#6a8db8', '#4a72a8', '#33547f', '#c0392b'], width=0.55)
ax.set_ylim(0.88, 0.95)
ax.set_ylabel('Validation Accuracy')
ax.set_title('Experiment Progression (train=6000 / val=1368, seed=42)')
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, a + 0.001, f'{a:.4f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_evolution.png'))
plt.close()

# ================= 图2: C 扫参曲线 (参数对比, 双面板: 5000/30000 特征) =================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
C1 = [0.01, 0.1, 1, 10, 100, 300, 1000]
lr_val_1 = [0.7982, 0.8480, 0.8991, 0.9167, 0.9189, 0.9123, 0.9123]
lr_tr_1 = [0.8518, 0.9017, 0.9647, 0.9993, 0.9998, 0.9998, 0.9998]
svm_val_1 = [0.1016, 0.8326, 0.9072, 0.9101, 0.9101, 0.9101, 0.9101]
ax1.semilogx(C1, lr_tr_1, 'o--', color='#b0b0b0', label='LR train')
ax1.semilogx(C1, lr_val_1, 'o-', color='#c0392b', label='LR val')
ax1.semilogx(C1, svm_val_1, 's-', color='#2e6da4', label='SVM(linear) val')
ax1.set_title('C sweep @ max_features=5000')
ax1.set_xlabel('C (log scale)'); ax1.set_ylabel('Accuracy')
ax1.legend(fontsize=8)
C2 = [10, 100, 300, 1000, 3000]
lr_val_2 = [0.9320, 0.9349, 0.9393, 0.9364, 0.9291]
ax2.semilogx(C2, lr_val_2, 'o-', color='#c0392b', label='LR val')
ax2.axvline(300, color='#7f8c8d', ls='--', lw=0.8)
ax2.text(320, 0.930, 'peak at C=300', fontsize=8, color='#7f8c8d')
ax2.set_title('C re-check @ max_features=30000\n(optimal C shifted 100 -> 300)')
ax2.set_xlabel('C (log scale)'); ax2.set_ylabel('Accuracy')
ax2.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_C_sweep.png'))
plt.close()

# ================= 图3: max_features 曲线 (参数对比) =================
mfs = [3000, 5000, 8000, 10000, 15000, 20000, 30000, 50000, 97284]
mfs_acc = [0.8882, 0.9189, 0.9254, 0.9276, 0.9320, 0.9320, 0.9349, 0.9349, 0.9349]
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.semilogx(mfs, mfs_acc, 'o-', color='#2e6da4')
ax.axvline(30000, color='#c0392b', ls='--', lw=1)
ax.text(33000, 0.890, 'chosen: 30000\n(long tail gives zero gain)', fontsize=8, color='#c0392b')
ax.set_xlabel('max_features (log scale)'); ax.set_ylabel('Validation Accuracy')
ax.set_title('Vocabulary Size Curve (LR C=100, ngram=(1,1))')
ax.set_ylim(0.87, 0.945)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_max_features.png'))
plt.close()

# ================= 图4: ngram 对比 (参数对比, 负结果) =================
mfs4 = [3000, 5000, 8000, 10000]
uni = [0.8882, 0.9189, 0.9254, 0.9276]
bi = [0.8772, 0.8962, 0.9174, 0.9196]
x = np.arange(len(mfs4))
w = 0.38
fig, ax = plt.subplots(figsize=(7, 3.6))
ax.bar(x - w/2, uni, w, label='ngram (1,1) unigram', color='#2e6da4')
ax.bar(x + w/2, bi, w, label='ngram (1,2) +bigram', color='#e08e45')
for i, (u, b) in enumerate(zip(uni, bi)):
    ax.text(i - w/2, u + 0.003, f'{u:.3f}', ha='center', fontsize=8)
    ax.text(i + w/2, b + 0.003, f'{b:.3f}', ha='center', fontsize=8)
ax.set_xticks(x); ax.set_xticklabels([f'{m}' for m in mfs4])
ax.set_xlabel('max_features'); ax.set_ylabel('Validation Accuracy')
ax.set_title('Bigram Hurts at Every Vocabulary Size (LR C=100)')
ax.legend(fontsize=8); ax.set_ylim(0.86, 0.94)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_ngram.png'))
plt.close()

# ================= 图6+7: 最终模型分类性能 (每类F1 + 混淆矩阵) =================
train_df = pd.read_csv(os.path.join(BASE_DIR, 'train_cleaned.csv'))
X = train_df['text'].astype(str).tolist()
y = train_df['target'].values
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, train_size=6000, stratify=y, random_state=42)
vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 1))
Xtr = vec.fit_transform(X_tr)
Xva = vec.transform(X_val)
final_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300,
                            random_state=42).fit(Xtr, y_tr)
y_pred = final_model.predict(Xva)

report = classification_report(y_val, y_pred, output_dict=True)
cls = [str(i) for i in range(10)]
f1 = [report[c]['f1-score'] for c in cls]
prec = [report[c]['precision'] for c in cls]
rec = [report[c]['recall'] for c in cls]
print(f'最终模型验证集准确率: {accuracy_score(y_val, y_pred):.4f}')
print(f'macro-F1: {report["macro avg"]["f1-score"]:.4f}')

x = np.arange(10)
w = 0.27
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(x - w, prec, w, label='Precision', color='#2e6da4')
ax.bar(x, f1, w, label='F1', color='#c0392b')
ax.bar(x + w, rec, w, label='Recall', color='#5aa35a')
ax.set_xticks(x); ax.set_xlabel('Class')
ax.set_ylabel('Score')
ax.set_title('Per-class Performance of Final Model (MLP(100,), 30000 feat)')
ax.legend(fontsize=8); ax.set_ylim(0.8, 1.0)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_per_class.png'))
plt.close()

cm = confusion_matrix(y_val, y_pred)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks(range(10)); ax.set_yticks(range(10))
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
for i in range(10):
    for j in range(10):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                fontsize=7, color='white' if cm[i, j] > cm.max()/2 else '#333333')
ax.set_title(f'Confusion Matrix (val 1368, {accuracy_score(y_val, y_pred):.4f})')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_confusion.png'))
plt.close()

# ================= 图7: 回帖率与类别 (清洗决策证据) =================
reply_rate = [48.8, 53.7, 47.6, 75.9, 81.8, 75.3, 77.8, 65.6, 69.8, 72.9]
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar([str(i) for i in range(10)], reply_rate, color='#8e6aa8')
ax.axhline(67.0, color='#c0392b', ls='--', lw=1, label='overall 67.0%')
ax.set_xlabel('Class'); ax.set_ylabel('Reply Rate (%)')
ax.set_title('"Re:" Reply Rate Varies Strongly by Class\n(evidence that Re: is a signal, not noise)')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_reply_rate.png'))
plt.close()

# ================= 图8: MLP 配置对比 (思考题2) =================
mlp_names = ['MLP\n(100,) it300', 'MLP\n(200,) it300', 'MLP\n(100,50) it300',
             'MLP\n(100,) it1000', 'LR\nC=300']
mlp_vals = [0.9401, 0.9415, 0.9342, 0.9401, 0.9393]
colors = ['#8fa8c8', '#8fa8c8', '#8fa8c8', '#8fa8c8', '#c0392b']
fig, ax = plt.subplots(figsize=(7, 3.6))
bars = ax.bar(mlp_names, mlp_vals, color=colors, width=0.55)
for b, v in zip(bars, mlp_vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.0006, f'{v:.4f}', ha='center', fontsize=9)
ax.set_ylim(0.925, 0.948)
ax.set_ylabel('Validation Accuracy')
ax.set_title('MLP Configurations vs LR (TF-IDF 30000)')
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_mlp.png'))
plt.close()

print('全部图表已保存到 figures/:')
for f in sorted(os.listdir(FIG_DIR)):
    print(' ', f)
