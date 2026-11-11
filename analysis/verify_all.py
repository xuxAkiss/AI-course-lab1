# -*- coding: utf-8 -*-
"""系统性差分验证脚本（verify_all.py）

原则：不信任实现，用**独立参考实现**逐题交叉验证。
- 题目一：A* 路径 vs BFS 最短路（独立实现）
- 题目二：A*（修正启发）代价 vs Dijkstra（h=0）代价
- 题目三：Alpha-Beta 值 vs Minimax 值 vs 按层归约的独立参考
- 题目四：搜索值 vs 独立递归穷举（随机宝藏价值分配，差分测试）

运行：python -X utf8 analysis/verify_all.py
"""

import heapq
import itertools
import os
import random
import sys
from collections import deque

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'solutions'))

DIRS4 = ((0, 1), (1, 0), (0, -1), (-1, 0))
PASS = True


def report(name, ok, detail=""):
    global PASS
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        PASS = False


# ================= 题目一：A* vs BFS =================
print("=" * 56)
print("题目一 差分验证：A* 路径 vs BFS 最短路（1000 张随机地图）")
print("=" * 56)
from p1_maze_astar import astar_maze  # noqa: E402


def bfs_dist(grid, start, goal):
    """独立参考：BFS 求最短步数；不可达返回 -1。"""
    rows, cols = len(grid), len(grid[0])
    dist = {start: 0}
    q = deque([start])
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return dist[(r, c)]
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in dist
                    and grid[nr][nc] != 'X'):
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return -1


random.seed(101)
fails = 0
for t in range(1000):
    size = random.randint(6, 15)
    p_obs = random.uniform(0.15, 0.35)
    g = [[('X' if random.random() < p_obs else '.') for _ in range(size)]
         for _ in range(size)]
    start = (0, 0)
    goal = (size - 1, size - 1)
    g[0][0] = 'S'
    g[size - 1][size - 1] = 'E'
    d_bfs = bfs_dist(g, start, goal)
    path = astar_maze(start, goal, g)
    if d_bfs == -1:
        ok = (path == [])
    else:
        ok = (len(path) - 1 == d_bfs and path[0] == start and path[-1] == goal
              and all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
                      for a, b in zip(path, path[1:]))
              and all(g[r][c] != 'X' for r, c in path))
    if not ok:
        fails += 1
        if fails <= 2:
            print(f"  失败地图 {size}×{size}: BFS={d_bfs}, A*={path}")
report("题目一 1000 张随机地图全部一致", fails == 0, f"failures={fails}")

# ================= 题目二：A* vs Dijkstra =================
print()
print("=" * 56)
print("题目二 差分验证：A*（修正启发） vs Dijkstra（1000 张随机地图）")
print("=" * 56)
from p2_weighted_terrain import astar_mincost  # noqa: E402

random.seed(202)
fails = 0
for t in range(1000):
    size = random.randint(5, 12)
    p_obs = random.uniform(0.1, 0.25)
    g = [[('X' if random.random() < p_obs else str(random.randint(1, 5)))
          for _ in range(size)] for _ in range(size)]
    g[0][0] = 'S'
    g[size - 1][size - 1] = 'E'
    c_ref, _, _ = astar_mincost((0, 0), (size - 1, size - 1), g, h_func=lambda a, b: 0)
    c_astar, _, _ = astar_mincost((0, 0), (size - 1, size - 1), g)
    if c_ref != c_astar:
        fails += 1
        if fails <= 2:
            print(f"  失败地图 {size}×{size}: Dijkstra={c_ref}, A*={c_astar}")
report("题目二 1000 张随机地图代价完全一致", fails == 0, f"failures={fails}")

# ================= 题目三：Alpha-Beta vs Minimax vs 独立参考 =================
print()
print("=" * 56)
print("题目三 差分验证：Alpha-Beta vs Minimax vs 独立按层归约（500 棵随机树）")
print("=" * 56)
from p3_alpha_beta import minimax, alpha_beta, random_tree, Stats  # noqa: E402
INF = float('inf')


def reference_regular_tree(root, depth, b):
    """独立参考实现：不用递归求值，收集叶子后按层做归约。"""
    leaves = []

    def collect(n):
        if not n.children:
            leaves.append(n.value)
        else:
            for c in n.children:
                collect(c)

    collect(root)
    vals = leaves
    level = depth
    while len(vals) > 1:
        level -= 1  # 父节点所在层；根(层0)是 MAX，偶数层 MAX
        take_max = (level % 2 == 0)
        vals = [(max if take_max else min)(vals[i:i + b])
                for i in range(0, len(vals), b)]
    return vals[0]


random.seed(303)
fails = 0
for t in range(500):
    depth = random.randint(2, 4)
    b = random.randint(2, 3)
    tree = random_tree(depth, b)
    v_mm = minimax(tree, True, Stats())
    v_ab = alpha_beta(tree, True, -INF, INF, Stats())
    v_ref = reference_regular_tree(tree, depth, b)
    if not (v_mm == v_ab == v_ref):
        fails += 1
        if fails <= 2:
            print(f"  失败树 depth={depth} b={b}: minimax={v_mm}, alpha_beta={v_ab}, ref={v_ref}")
report("题目三 500 棵随机树三实现完全一致", fails == 0, f"failures={fails}")

# ================= 题目四：搜索 vs 独立穷举（随机价值差分） =================
print()
print("=" * 56)
print("题目四 差分验证：Alpha-Beta 搜索 vs 独立递归穷举（100 组随机价值 × H∈{2,4}）")
print("=" * 56)
import p4_treasure_game as p4  # noqa: E402

names = ['A', 'B', 'T1', 'T2', 'T3', 'T4']
dl = p4.distance_matrix()
dmat = {p: {q: dl[i][j] for j, q in enumerate(names)} for i, p in enumerate(names)}


def brute_minimax(remaining, turn, pA, pB, U, values, depth, horizon):
    """独立参考：递归穷举博弈树（不剪枝、不复用 p4 任何函数），支持深度 H。"""
    if depth >= horizon or not remaining:
        return U
    results = []
    for t in remaining:
        if turn == 'A':
            results.append(brute_minimax(remaining - {t}, 'B', t, pB,
                                         U + values[t] - dmat[pA][t],
                                         values, depth + 1, horizon))
        else:
            results.append(brute_minimax(remaining - {t}, 'A', pA, t,
                                         U - (values[t] - dmat[pB][t]),
                                         values, depth + 1, horizon))
    return max(results) if turn == 'A' else min(results)


def brute_root_moves(values, horizon):
    """独立参考：根节点各首步的精确博弈值。"""
    moves = {}
    for first in sorted(values):
        g = values[first] - dmat['A'][first]
        moves[first] = brute_minimax(set(values) - {first}, 'B', first, 'B',
                                     g, values, 1, horizon)
    return moves


random.seed(404)
fails = 0
orig_values = dict(p4.TREASURE_VALUES)
try:
    # 先用题目原值验证 H=2 与 H=4 的全部明细（对照独立穷举）
    for horizon in (2, 4):
        _, _, mv = p4.alpha_beta_root(
            p4.GameState(pos={'A': 'A', 'B': 'B'}, remaining=set(p4.TREASURE_VALUES),
                         turn='A', depth=0), horizon, dmat, p4.SearchStats())
        mv_ref = brute_root_moves(p4.TREASURE_VALUES, horizon)
        if mv != mv_ref:
            fails += 1
            print(f"  [原值 H={horizon}] 首步明细不符: {mv} vs 穷举 {mv_ref}")

    # 随机价值差分：比较**每个首步的精确值**（不只是最大值）
    for t in range(100):
        perm = random.sample([6, 7, 7, 9, 5, 8], 4)  # 每轮随机价值（可能有并列）
        values = dict(zip(['T1', 'T2', 'T3', 'T4'], perm))
        p4.TREASURE_VALUES = values  # 猴子补丁：替换 p4 的模块级价值表

        root = p4.GameState(pos={'A': 'A', 'B': 'B'}, remaining=set(values),
                            turn='A', depth=0)
        for horizon in (2, 4):
            _, _, mv = p4.alpha_beta_root(root, horizon, dmat, p4.SearchStats())
            mv_ref = brute_root_moves(values, horizon)
            if mv != mv_ref:
                fails += 1
                if fails <= 3:
                    print(f"  失败 H={horizon} 价值{values}: 搜索={mv}, 穷举={mv_ref}")
finally:
    p4.TREASURE_VALUES = orig_values
report("题目四 原值明细 + 100 组随机价值 × H∈{2,4} 全部首步值一致",
       fails == 0, f"failures={fails}")

# ================= 汇总 =================
print()
print("=" * 56)
print("总结果：" + ("全部通过 ✓" if PASS else "存在失败 ✗"))
print("=" * 56)
sys.exit(0 if PASS else 1)
