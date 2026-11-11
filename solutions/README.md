# 实验二：搜索算法 — 代码说明（README）

当代人工智能实验课 · 实验二 · A* 与 Alpha-Beta 剪枝
提交截止：2026-11-12 23:59:59

## 环境要求

- Python 3.7+（仅用标准库 `heapq`、`itertools`、`random`、`typing`，**无第三方依赖**）
- Windows 终端运行建议加 `-X utf8`（避免 GBK 控制台打印中文报错）：
  ```bash
  python -X utf8 p1_maze_astar.py
  ```

## 文件与运行方式

| 文件 | 题目 | 运行命令 |
|---|---|---|
| `p1_maze_astar.py` | 题一：迷宫寻路（A*） | `python -X utf8 p1_maze_astar.py` |
| `p2_weighted_terrain.py` | 题二：加权地形（A*） | `python -X utf8 p2_weighted_terrain.py` |
| `p3_alpha_beta.py` | 题三：Alpha-Beta 剪枝 | `python -X utf8 p3_alpha_beta.py` |
| `p4_treasure_game.py` | 题四：竞争宝藏分配（综合） | `python -X utf8 p4_treasure_game.py` |

每个文件都是**自包含**的：直接运行时输出 PDF 测试用例的结果与断言校验（全部通过 ✓），并打印对照实验所需统计量（展开节点数、叶子访问数、搜索节点数）。

## 对外接口（与 PDF 输入输出格式一致）

```python
# 题一：返回最短路径坐标列表；无解返回 []
from p1_maze_astar import astar_maze
astar_maze((0, 0), (3, 4), grid)          # grid: List[List[str]]

# 题二：返回最小总代价 (int)；无解返回 -1
from p2_weighted_terrain import min_move_cost
min_move_cost((0, 0), (4, 4), grid)

# 题三：返回 MAX 的最佳得分（TreeNode 根节点）
from p3_alpha_beta import alpha_beta, alpha_beta_json, TreeNode
alpha_beta(root)                           # 返回 int
alpha_beta_json(json_tree["tree_structure"])  # JSON 输入格式版本

# 题四：返回 (最优首步, 最优收益差)
from p4_treasure_game import distance_matrix, best_first_move
dist = distance_matrix()                   # 6×6 距离矩阵（A,B,T1~T4）
best_first_move(4, dist, 'alpha_beta')     # → ('T3', 3)
```

## 测试用例与预期输出（均通过断言校验）

| 题目 | 输入 | 预期输出 | 实测 |
|---|---|---|---|
| 一 | (0,0)→(3,4)，5×5 网格 | `[(0,0),(0,1),(0,2),(1,2),(1,3),(1,4),(2,4),(3,4)]` | 逐格一致 ✓ |
| 二 | (0,0)→(4,4)，加权网格 | `7` | 7 ✓ |
| 三 | PDF 示例博弈树 | `5` | 5 ✓ |
| 四 | H=2 | `{"best_target": "T2", "best_value": -1}` | T2, -1 ✓ |
| 四 | H=4 | `{"best_target": "T3", "best_value": 3}` | T3, 3 ✓ |

## 结果可复现性

- 所有随机实验（题一随机地图、题三随机树）均固定 `random.seed(...)`，重复运行结果一致；
- 对照实验数据汇总脚本：`python -X utf8 ../analysis/collect_report_data.py`，输出写入 `analysis/report_data.txt`；
- 平局处理规则（题一：f 相同时优先展开 g 更大的节点；方向序 右→下→左→上）已固定，路径输出稳定可复现。

## 代码结构约定

- 每题一个自包含文件，对外暴露一个"题目要求格式"的函数（上表）+ 一个带统计的通用搜索函数（供对照实验）；
- 统计信息（展开节点数 / 开放列表峰值 / 叶子访问数 / 博弈树节点数）随搜索结果返回，支撑报告中的实验数据分析；
- 文件头 docstring 记录题目原文规则、输入输出格式与关键设计决策。
