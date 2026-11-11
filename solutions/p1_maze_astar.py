# -*- coding: utf-8 -*-
"""题目一：迷宫寻路（A* 算法）

在二维网格中用 A* 算法找到起点 S 到终点 E 的最短路径。
网格中 '.' / 'S' / 'E' 为可通过路径，'X' 为不可通过的障碍物。

规则
----
- 移动：只能上下左右四个方向移动。
- 代价：每移动一格的代价（g-cost）为 1。
- 启发函数 h(n)：曼哈顿距离 |Δrow| + |Δcol|（四方向移动下所需步数的下界）。

输入格式
--------
- start : Tuple[int, int]   起点坐标 (row, col)
- goal  : Tuple[int, int]   终点坐标 (row, col)
- grid  : List[List[str]]   地图（等效二维数组）

输出格式
--------
- List[Tuple[int, int]]：按顺序包含起点到终点最短路径上的所有坐标；
  无解时返回空列表 []。
- 存在多条等长最短路径时，输出任意一条即可。

参考：实验二要求_new.pdf 第 3-5 页
"""

import heapq
from typing import Callable, List, Tuple

Grid = List[List[str]]
Path = List[Tuple[int, int]]

# 四个移动方向：右、下、左、上。
# 该顺序配合堆的平局规则（f 相同时优先展开 g 更大的节点），
# 恰好输出与 PDF 预期一致的那条最短路径（见学习文档"平局处理"一节）。
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def heuristic_manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """曼哈顿距离：每步代价为 1 时，从 a 到 b 所需步数的下界（可采纳且一致）。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def heuristic_zero(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    """恒为 0 的启发函数：A* 退化为 Dijkstra / 广度优先搜索（用于对照实验）。"""
    return 0


def astar_search(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: Grid,
    h_func: Callable[[Tuple[int, int], Tuple[int, int]], int] = heuristic_manhattan,
) -> Tuple[Path, dict]:
    """通用 A* 搜索。

    返回 (路径, 统计信息)；无解时路径为 []。
    统计信息（expanded=展开节点数、max_open=开放列表峰值）供实验报告对照分析使用。

    平局规则：f 相同时优先展开 g 更大（离起点更远）的节点——能更快逼近终点，
    且启发函数一致时不影响最优性。
    """
    rows, cols = len(grid), len(grid[0])

    def passable(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols and grid[r][c] != 'X'

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

        if cur == goal:  # 目标被弹出时，g(goal) 必为最短步数
            path = [cur]
            while cur in came_from:  # 沿 came_from 回溯重建路径
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            stats["path_len"] = len(path) - 1  # 移动步数
            return path, stats

        for dr, dc in DIRS:
            nb = (cur[0] + dr, cur[1] + dc)
            if not passable(*nb) or nb in closed:
                continue
            ng = g[cur] + 1  # g-cost 每步为 1
            if nb not in g or ng < g[nb]:
                g[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_heap, (ng + h_func(nb, goal), -ng, seq, nb))
                seq += 1

    return [], stats  # 开放列表耗尽：无解


def astar_maze(start: Tuple[int, int], goal: Tuple[int, int], grid: Grid) -> Path:
    """题目一要求的对外接口：返回最短路径坐标列表，无解返回 []。"""
    path, _ = astar_search(start, goal, grid, heuristic_manhattan)
    return path


if __name__ == "__main__":
    # ---------- PDF 测试用例 ----------
    start, goal = (0, 0), (3, 4)
    grid = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.'],
    ]
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

    path, stats = astar_search(start, goal, grid, heuristic_manhattan)
    print("题目一 测试用例")
    print("路径:", path)
    print("与 PDF 预期输出一致:", path == expected)
    print("统计: 展开节点数 =", stats["expanded"], "，开放列表峰值 =", stats["max_open"])

    # ---------- 对照实验：h=0（等价 Dijkstra/BFS） ----------
    _, stats0 = astar_search(start, goal, grid, heuristic_zero)
    print("对照 h=0: 展开节点数 =", stats0["expanded"])

    # ---------- 无解用例 ----------
    grid_no = [['S', 'X'], ['X', 'E']]
    print("无解用例:", astar_maze((0, 0), (1, 1), grid_no))

    # ---------- 自动校验：路径合法且最短 ----------
    def validate(p: Path) -> bool:
        if not p or p[0] != start or p[-1] != goal:
            return False
        for (r1, c1), (r2, c2) in zip(p, p[1:]):
            if abs(r1 - r2) + abs(c1 - c2) != 1 or grid[r2][c2] == 'X':
                return False
        return len(p) - 1 == abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    assert validate(path), "路径校验失败"
    assert astar_maze((0, 0), (1, 1), grid_no) == []
    print("全部断言通过 ✓")
