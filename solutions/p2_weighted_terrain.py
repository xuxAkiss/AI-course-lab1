# -*- coding: utf-8 -*-
"""题目二：加权地形（A* 算法）

地图由代表移动代价的数字组成（1=平原代价 1，3=沼泽代价 3，…）。
用 A* 计算从起点 S 到终点 E 的最小移动代价。

规则
----
- 移动：只能上下左右四个方向。
- 代价：从节点 A 到相邻节点 B 的代价 = 进入节点 B 的地形权值。
- ★ 关键规则：进入终点 E 的代价视为 0（PDF 原话："最后一步没有代价，
  即可以将 E 视为 0"）。

启发函数（重要）
----------------
PDF 提示："这时候使用曼哈顿距离仍然是一个可采纳的启发函数吗？"

答案：若直接把 h(n) 设为 曼哈顿距离 × 最小地形代价 min_cost，会高估剩余代价——
因为最后一步进入 E 是免费的，真实剩余代价 ≥ min_cost × (d(n,goal) − 1)，
而不是 min_cost × d。
因此本实现取 h(n) = min_cost × max(0, d(n,goal) − 1)，可采纳且一致
（详细证明见 docs/学习文档-题目二）。当 min_cost = 1 时即退化为"曼哈顿减一"。

输入格式
--------
- start / goal : Tuple[int, int]；grid : List[List[str]]（等效二维数组）。

输出格式
--------
- int：起点到终点的最小总代价；无解返回 -1。

参考：实验二要求_new.pdf 第 5-7 页
"""

import heapq
from typing import Callable, List, Optional, Tuple

Grid = List[List[str]]
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def enter_cost(grid: Grid, r: int, c: int) -> Optional[int]:
    """进入格子 (r,c) 的代价；返回 None 表示不可通行。"""
    v = grid[r][c]
    if isinstance(v, int):
        return v
    if v == 'X':
        return None
    if v == 'E':
        return 0  # ★ 关键规则：进入 E 不计代价
    if v == 'S':
        return 0  # 起点不会被再次进入，设 0 仅为安全
    if v.isdigit():
        return int(v)
    return None  # 未知符号一律视为不可通行


def min_terrain_cost(grid: Grid) -> int:
    """地图上数字地形的最小代价（启发函数缩放用）；无数字地形时返回 1。"""
    costs = [int(v) for row in grid for v in row if isinstance(v, str) and v.isdigit()]
    return min(costs) if costs else 1


def heuristic_admissible(node: Tuple[int, int], goal: Tuple[int, int], min_cost: int) -> int:
    """h(n) = min_cost × max(0, 曼哈顿距离 − 1)。

    理由：最后一步进入 E 免费，剩余 d 步中最多 d−1 步需要付费，
    每步 ≥ min_cost ⇒ 真实剩余代价 ≥ min_cost × (d−1)，h 不会高估（可采纳）；
    且对任意一条边 |h(n)−h(n')| ≤ min_cost ≤ 边代价（一致），
    因此 A* 弹出一个节点时其 g 已是最终最短代价，无需重新展开。
    """
    d = abs(node[0] - goal[0]) + abs(node[1] - goal[1])
    return min_cost * max(0, d - 1)


def astar_mincost(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: Grid,
    h_func: Optional[Callable[[Tuple[int, int], Tuple[int, int]], int]] = None,
) -> Tuple[int, dict, Optional[List[Tuple[int, int]]]]:
    """通用加权 A*。返回 (最小总代价, 统计信息, 路径或 None)。

    - h_func 为 None 时使用默认可采纳启发函数 heuristic_admissible；
    - 传 lambda a, b: 0 即退化为 Dijkstra，用于对照实验。
    """
    rows, cols = len(grid), len(grid[0])
    if start == goal:
        return 0, {"expanded": 0, "max_open": 0}, [start]

    mc = min_terrain_cost(grid)
    if h_func is None:
        h_func = lambda a, b: heuristic_admissible(a, b, mc)

    open_heap = []  # 开放列表（小顶堆），元素 (f, -g, 序号, 坐标)
    seq = 0
    g = {start: 0}
    came_from = {}
    heapq.heappush(open_heap, (g[start] + h_func(start, goal), -g[start], seq, start))
    seq += 1
    closed = set()
    stats = {"expanded": 0, "max_open": 1}

    while open_heap:
        stats["max_open"] = max(stats["max_open"], len(open_heap))
        f, neg_g, _, cur = heapq.heappop(open_heap)
        if cur in closed:
            continue
        closed.add(cur)
        stats["expanded"] += 1

        if cur == goal:  # 目标被弹出时，g(goal) 必为最小总代价
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return g[goal], stats, path

        r, c = cur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            step = enter_cost(grid, nr, nc)
            if step is None:
                continue
            nb = (nr, nc)
            if nb in closed:
                continue
            ng = g[cur] + step
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h_func(nb, goal), -ng, seq, nb))
                seq += 1

    return -1, stats, None  # 开放列表耗尽：无解


def min_move_cost(start: Tuple[int, int], goal: Tuple[int, int], grid: Grid) -> int:
    """题目二要求的对外接口：返回最小总代价，无解返回 -1。"""
    cost, _, _ = astar_mincost(start, goal, grid)
    return cost


if __name__ == "__main__":
    # ---------- PDF 测试用例 ----------
    start, goal = (0, 0), (4, 4)
    grid = [
        ['S', '1', '1', '8', '1'],
        ['1', '8', '3', '1', '1'],
        ['1', '8', '1', '8', '1'],
        ['1', '1', '1', '8', '1'],
        ['8', '8', '1', '1', 'E'],
    ]
    cost, stats, path = astar_mincost(start, goal, grid)
    print("题目二 测试用例")
    print("最小总代价:", cost, "（PDF 预期输出 7:", cost == 7, "）")
    print("路径:", path)
    print("统计: 展开节点数 =", stats["expanded"], "，开放列表峰值 =", stats["max_open"])

    # ---------- 路径代价的累计过程（报告素材） ----------
    print("\n路径代价累计:")
    total = 0
    for i, (r, c) in enumerate(path):
        if i == 0:
            print(f"  第{i}步: 起点 ({r},{c})，累计代价 0")
        else:
            step = enter_cost(grid, r, c)
            total += step
            print(f"  第{i}步: 进入 ({r},{c}) 代价 {step}，累计代价 {total}")

    # ---------- 对照实验：h=0（等价 Dijkstra） ----------
    cost0, stats0, _ = astar_mincost(start, goal, grid, h_func=lambda a, b: 0)
    print(f"\n对照 h=0: 代价 {cost0}，展开节点数 {stats0['expanded']}")

    # ---------- 无解用例 ----------
    grid_no = [['S', 'X'], ['X', 'E']]
    print("无解用例:", min_move_cost((0, 0), (1, 1), grid_no))

    # ---------- 断言 ----------
    assert cost == 7, "测试用例代价应为 7"
    assert min_move_cost((0, 0), (1, 1), grid_no) == -1
    print("全部断言通过 ✓")
