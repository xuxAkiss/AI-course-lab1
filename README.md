# 实验一：文本分类

新闻文本 10 分类任务。最终方案：**邮件头剥离 + TF-IDF(30000, unigram) + MLP 单隐层(100 神经元)**，
验证集准确率 94.01%（5 种子复验均值 94.28% ± 0.29%）。

## 运行环境

- Windows 11 + conda 环境 `project1`（创建命令见下）
- Python 3.10.21 / numpy 2.2.5 / pandas 2.3.3 / scikit-learn 1.7.1
- matplotlib 3.10.9（仅 `plot_analysis.py` 生成报告图表需要，`main.py` 不依赖）

```bash
conda create -n project1 python=3.10
conda activate project1
conda install numpy pandas scikit-learn
```

## 复现步骤

```bash
cd lab1
python main.py        # 一步到位: 从原始 CSV 清洗 -> 训练 -> 生成 predictions.csv
```

`main.py` 会输出：训练集/测试集条数、词表大小、训练集自检准确率、预测类别分布，
并生成提交文件 `predictions.csv`（2457 行、无表头、单列，格式与示例一致）。

## 项目结构

| 文件 | 作用 |
|------|------|
| `main.py` | **提交主脚本**：最终配置全量重训 + 生成 predictions.csv |
| `preprocess.py` | 数据清洗（剥邮件头等，步骤均可开关） |
| `baseline.py` | 基线：TF-IDF 5000 + 朴素贝叶斯（验证 89.84%） |
| `tune_C.py` | C 值扫参（SVM 线性核 vs 逻辑回归，对数间隔） |
| `stopwords_test.py` | 停用词实验（文档频率统计 + 开关对比） |
| `grid_search.py` | max_features × ngram_range 网格扫参 |
| `finalize_features.py` | 词表曲线收尾 + C 值重验 |
| `inspect_weights.py` | 模型可解释性：每类权重最高的词 |
| `evaluate_cleaned.py` | 清洗效果评估（噪声词残留 + 验证集对比） |
| `mlp.py` | MLP 结构/轮数对比实验（思考题 2） |
| `split_sensitivity.py` | 5 种子划分敏感性复验（支持断点续跑） |
| `loss_curve.py` | 最终模型逐轮训练损失曲线（报告图 5，原始数据存 loss_curve.csv） |
| `plot_analysis.py` | 生成 figures/ 下全部报告图表 |
| `实验决策记录.md` | **全部实验决策与理由**（清洗、划分、调参、负结果） |
| `实验报告.md` | 正式实验报告（约 2 页 PDF 量，引 figures/ 图表） |
| `实验报告.docx` | Word 版报告（build_docx.py 生成，图 3/4 并排排版） |
| `build_docx.py` | Word 报告生成脚本（FIG_W 字典可调图片大小控制页数） |
| `学习文档.md` | 原理讲解文档（每个步骤为什么这么做） |
| `train_data.csv` / `test_data_unlabeled.csv` | 原始数据（未改动） |
| `train_cleaned.csv` / `test_cleaned.csv` | 清洗后数据（preprocess.py 生成） |

## 最终配置（锁定过程见 实验决策记录.md）

| 项目 | 选择 |
|------|------|
| 数据清洗 | 剥离邮件头（保留 Subject 行与 Re: 标记——实测为类别信号，决策11） |
| 特征 | TF-IDF，max_features=30000，ngram=(1,1) |
| 模型 | MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)（可用 MODEL 开关切回 LR C=300） |
| 随机种子 | 42（全局固定） |
| 验证集 | 固定划分 6000/1368，分层抽样，仅用于选型，最终模型用全量 7368 条重训 |

## 实验路线（验证集准确率）

| 阶段 | 配置 | 准确率 |
|------|------|--------|
| 基线 | 朴素贝叶斯 | 89.84% |
| 模型+调参 | LR C=100 (5000词) | 91.89% |
| 特征扩展 | LR C=100 (30000词) | 93.49% |
| 正则化重验 | LR C=300 (30000词) | 93.93% |
| 多种子复验换模型 | **MLP(100,) (30000词)** | **94.01%（5种子均值 94.28±0.29）** |
