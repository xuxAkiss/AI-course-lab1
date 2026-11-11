# 实验二：搜索算法（A\* 与 Alpha-Beta 剪枝）

当代人工智能实验课 · 实验二。四道题全部用**纯 Python 标准库**实现，无第三方依赖，
每个题目文件自包含、可直接运行并自带断言校验。

> 本分支（`lab2`）只存放实验二内容，实验一在 `main` 分支。

| 题目 | 内容 | 文件 |
|---|---|---|
| 题一 | 迷宫寻路（A\*，曼哈顿启发，四方向） | `solutions/p1_maze_astar.py` |
| 题二 | 加权地形最小代价路径（A\*，修正启发） | `solutions/p2_weighted_terrain.py` |
| 题三 | 博弈树 Alpha-Beta 剪枝 | `solutions/p3_alpha_beta.py` |
| 题四 | 竞争宝藏分配（A\* + 对抗搜索综合） | `solutions/p4_treasure_game.py` |

## 环境

- Python 3.7+（实测 3.14.7），仅用 `heapq` / `itertools` / `random` / `typing` 等标准库
- Windows 终端建议加 `-X utf8`，避免 GBK 控制台打印中文时报错

无需 `pip install`，克隆后即可运行。

## 目录结构

```
.
├── README.md                  # 本文件：运行与复现说明
├── solutions/
│   ├── README.md              # 各题接口、测试用例与预期输出
│   ├── p1_maze_astar.py       # 题一
│   ├── p2_weighted_terrain.py # 题二
│   ├── p3_alpha_beta.py       # 题三
│   └── p4_treasure_game.py    # 题四
└── analysis/
    └── verify_all.py          # 独立参考实现交叉验证（差分测试）
```

## 快速开始

```bash
cd solutions
python -X utf8 p1_maze_astar.py   # 迷宫 A*：输出路径 + 与题目预期比对 + 断言校验
python -X utf8 p2_weighted_terrain.py
python -X utf8 p3_alpha_beta.py
python -X utf8 p4_treasure_game.py
```

每个脚本运行后会打印测试用例结果与统计量（展开节点数 / 开放列表峰值 / 叶子访问数），
末尾输出 `全部断言通过 ✓` 表示该题正确。

## 一键复现实验结论

`analysis/verify_all.py` 用**独立参考实现**对四道题做差分验证（不信任被测代码），
在仓库根目录执行：

```bash
python -X utf8 analysis/verify_all.py
```

- 题一：A\* 路径 vs BFS 最短路，1000 张随机地图
- 题二：A\*（修正启发）代价 vs Dijkstra（h=0），1000 张随机地图
- 题三：Alpha-Beta 值 vs Minimax 值 vs 按层归约的独立参考，500 棵随机树
- 题四：搜索值 vs 独立递归穷举，100 组随机宝藏价值 × H ∈ {2, 4}

预期最后一行输出 `总结果：全部通过 ✓`（本机实测 4/4 项 PASS）。

## 对外接口（与题目 PDF 输入输出格式一致）

```python
from p1_maze_astar import astar_maze
astar_maze((0, 0), (3, 4), grid)              # → 最短路径坐标列表；无解返回 []

from p2_weighted_terrain import min_move_cost
min_move_cost((0, 0), (4, 4), grid)           # → 最小总代价 int；无解返回 -1

from p3_alpha_beta import alpha_beta, TreeNode
alpha_beta(root)                              # → MAX 的最佳得分 int

from p4_treasure_game import best_first_move
best_first_move(4, distance_matrix())         # → (最优首步, 最优收益差)
```

（在 `solutions/` 目录下 `import`，或把该目录加入 `PYTHONPATH`。）

## 可复现性说明

- 所有随机实验（随机地图、随机博弈树、随机宝藏价值）均固定随机种子，重复运行结果一致；
- A\* 的平局处理规则已固定（f 相同时优先展开 g 更大的节点，方向序 右→下→左→上），
  因此题一输出的最短路径稳定为题目预期的那一条；
- 题二启发函数按“进入格代价 ≥ 1”修正为曼哈顿距离 × 最小地形代价，保证可采纳性，
  与 Dijkstra 结果逐例一致。