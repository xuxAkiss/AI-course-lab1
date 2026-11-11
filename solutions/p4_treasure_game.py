# -*- coding: utf-8 -*-
"""题目四：竞争环境下的宝藏分配（综合题：A* + Alpha-Beta 剪枝）

两名玩家 A（MAX）与 B（MIN）轮流争夺四个宝藏，每回合选一个尚未获得的宝藏，
沿最小代价路径移动到该位置。用 A* 计算距离，用 Minimax / Alpha-Beta
在给定搜索深度 H 下求 A 的最优首步。

地图（X 障碍；数字 = 进入该格代价；A/B/T 所在格进入代价为 1）
      0   1   2   3   4   5   6
 0    A   1   1   X   2   1   T1
 1    1   X   6   X   1   X   1
 2    1   1   T2  1   1   1   1
 3    X   1   X   8   X   1   X
 4    1   1   1   1   T3  1   1
 5    1   X   9   X   1   X   1
 6    T4  1   1   X   1   1   B

宝藏价值：T1=9，T2=6，T3=7，T4=7
（PDF 表格中 T2、T3 的值因排版被打乱到页尾，此处经预期输出反推验证）

规则
----
- 行动顺序 A → B → A → B，共四回合；每回合只选一个宝藏，获取后更新玩家位置。
- 经过其他宝藏不自动收集；已获取宝藏的位置仍可通行；另一玩家不阻挡寻路。
- Gain(p, Ti) = Vi − d(p, Ti)（d 为最小移动代价）；
  U = Score_A − Score_B；A 最大化 U，B 最小化 U；
  到达搜索深度或宝藏取完时，以当前 U 作为评价值。

任务要求（PDF 第 15 页）
------------------------
1. 用 A* 计算 A、B、T1~T4 两两之间的最小移动代价，输出 6×6 距离矩阵；
2. 自行设计博弈状态（至少记录双方位置、剩余宝藏、累计收益、当前玩家）；
3. 分别使用 Minimax 和 Alpha-Beta，在深度 H=2、H=4 时计算 A 的最优首步与最优收益差。

预期输出：H=2 → {"best_target": "T2", "best_value": -1}
          H=4 → {"best_target": "T3", "best_value": 3}

参考：实验二要求_new.pdf 第 14-15 页
"""

import heapq
import itertools
from typing import Dict, List, Optional, Tuple

INF = float("inf")

# ---------------- 地图与宝藏定义 ----------------

# 地图：'X' 障碍；数字字符串为进入代价；A/B/T 所在格进入代价为 1
GRID = [
    ['A', '1', '1', 'X', '2', '1', 'T1'],
    ['1', 'X', '6', 'X', '1', 'X', '1'],
    ['1', '1', 'T2', '1', '1', '1', '1'],
    ['X', '1', 'X', '8', 'X', '1', 'X'],
    ['1', '1', '1', '1', 'T3', '1', '1'],
    ['1', 'X', '9', 'X', '1', 'X', '1'],
    ['T4', '1', '1', 'X', '1', '1', 'B'],
]

# 六个关键点：A、B 初始位置与四个宝藏位置
POINTS: Dict[str, Tuple[int, int]] = {
    'A': (0, 0), 'B': (6, 6),
    'T1': (0, 6), 'T2': (2, 2), 'T3': (4, 4), 'T4': (6, 0),
}

# 宝藏价值（PDF 表格；T2、T3 的位置经验证）
TREASURE_VALUES: Dict[str, int] = {'T1': 9, 'T2': 6, 'T3': 7, 'T4': 7}

DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def enter_cost(r: int, c: int) -> Optional[int]:
    """进入格子 (r,c) 的代价；返回 None 表示不可通行。

    进入代价制：数字地形按数字；A/B/T 所在格代价为 1；X 为障碍。
    （与题目二不同：此处 B、T 都是普通可行走格，没有"终点免费"规则。）
    """
    v = GRID[r][c]
    if v == 'X':
        return None
    if v.isdigit():
        return int(v)
    return 1  # 'A' / 'B' / 'T1'~'T4'


# ---------------- 第一步：A* 计算两两最小移动代价 ----------------

def astar_min_cost(start: str, goal: str) -> int:
    """用 A* 计算从关键点 start 到 goal 的最小移动代价。

    启发函数 h(n) = 到此为止的最小地形代价 × 曼哈顿距离
    （网格中最小进入代价为 1，曼哈顿距离 × 1 为可采纳下界）。
    注意：起点格的进入代价不计（我们已经在格子上），实现时把起点视作 0 代价。
    """
    sr, sc = POINTS[start]
    gr, gc = POINTS[goal]
    if start == goal:
        return 0

    def h(r: int, c: int) -> int:
        # 全图最小进入代价为 1，曼哈顿距离即代价下界（可采纳）
        return abs(r - gr) + abs(c - gc)

    rows, cols = len(GRID), len(GRID[0])
    open_heap = [(0 + h(sr, sc), 0, (sr, sc))]
    g: Dict[Tuple[int, int], int] = {(sr, sc): 0}
    closed = set()

    while open_heap:
        f, gcur, (r, c) = heapq.heappop(open_heap)
        if (r, c) in closed:
            continue
        closed.add((r, c))
        if (r, c) == (gr, gc):
            return gcur
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            cost = enter_cost(nr, nc)
            if cost is None:
                continue
            ng = gcur + cost
            if (nr, nc) not in g or ng < g[(nr, nc)]:
                g[(nr, nc)] = ng
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))
    return -1  # 不可达


def distance_matrix() -> List[List[int]]:
    """输出 6×6 距离矩阵（顺序：A、B、T1、T2、T3、T4）。

    本题宝藏/玩家位置固定且已获取宝藏仍可通行、玩家不阻挡寻路，
    因此两两距离是常数 —— 只需在开局预计算一次，博弈过程中直接查表。
    """
    names = ['A', 'B', 'T1', 'T2', 'T3', 'T4']
    return [[astar_min_cost(p, q) for q in names] for p in names]


# ---------------- 第二步：博弈状态设计 ----------------

class GameState:
    """博弈状态：双方位置、剩余宝藏、累计收益差、当前玩家。

    - pos['A'] / pos['B']：双方当前所在位置（'A'、'B' 或已获取的宝藏名）；
    - remaining：尚未被获取的宝藏名集合；
    - score：累计收益差 U = Score_A − Score_B（A 取宝 +Gain，B 取宝 −Gain）；
    - turn：轮到谁行动，'A'（MAX）或 'B'（MIN）；
    - depth：已经进行的回合数（每回合一个玩家取一个宝藏）。
    """

    __slots__ = ('pos', 'remaining', 'score', 'turn', 'depth')

    def __init__(self, pos, remaining, score=0, turn='A', depth=0):
        self.pos = pos              # {'A': 位置名, 'B': 位置名}
        self.remaining = remaining  # frozenset 或 set，剩余宝藏名
        self.score = score          # 累计收益差 U
        self.turn = turn            # 当前玩家
        self.depth = depth          # 已行动回合数


def apply_move(state: GameState, target: str, dist: Dict[str, Dict[str, int]]) -> GameState:
    """执行一步：当前玩家移动到 target，获取该宝藏，返回新状态。"""
    mover = state.turn
    gain = TREASURE_VALUES[target] - dist[state.pos[mover]][target]
    new_pos = dict(state.pos)
    new_pos[mover] = target  # 玩家位置更新到宝藏处
    new_score = state.score + (gain if mover == 'A' else -gain)  # A 加、B 减
    return GameState(
        pos=new_pos,
        remaining=state.remaining - {target},
        score=new_score,
        turn='B' if mover == 'A' else 'A',
        depth=state.depth + 1,
    )


# ---------------- 第三步：Minimax 与 Alpha-Beta ----------------

class SearchStats:
    """统计搜索开销：展开的博弈树节点数（供报告对照分析）。"""

    def __init__(self):
        self.nodes = 0


def minimax(state: GameState, horizon: int, dist, stats: Optional[SearchStats] = None) -> float:
    """朴素 Minimax：不剪枝。

    深度语义：H 表示"从根节点起最多搜索的层数"（根节点深度为 0）。
    即 H=2 → 搜索 A、B 各一层后停止评估；H=4 → 搜索 A、B、A、B 四层即终局。
    （该语义经预期输出 H=2→-1、H=4→3 反推确认。）

    终止条件：state.depth >= horizon 或宝藏取完，
    以当前累计收益差 U 作为评价值（即 state.score）。
    A（MAX）最大化 U，B（MIN）最小化 U。
    """
    if stats is not None:
        stats.nodes += 1
    if state.depth >= horizon or not state.remaining:
        return state.score  # 到达深度或宝藏取完：以当前 U 评价

    if state.turn == 'A':
        return max(minimax(apply_move(state, t, dist), horizon, dist, stats)
                   for t in sorted(state.remaining))
    else:
        return min(minimax(apply_move(state, t, dist), horizon, dist, stats)
                   for t in sorted(state.remaining))


def minimax_root(state: GameState, horizon: int, dist, stats: Optional[SearchStats] = None):
    """根节点版 Minimax：返回 (最优首步, 最优值, 各首步值)。

    根即 A（MAX）的一层，depth 从 0 计——与 horizon 语义一致。
    """
    if stats is not None:
        stats.nodes += 1
    stats0 = stats if stats is not None else SearchStats()
    best_target, best_value, all_moves = None, -INF, {}
    if state.depth >= horizon or not state.remaining:
        return None, state.score, {}
    for t in sorted(state.remaining):
        value = minimax(apply_move(state, t, dist), horizon, dist, stats0)
        all_moves[t] = value
        if value > best_value:
            best_value, best_target = value, t
    return best_target, best_value, all_moves


def alpha_beta(
    state: GameState,
    horizon: int,
    dist,
    alpha: float = -INF,
    beta: float = INF,
    stats: Optional[SearchStats] = None,
) -> float:
    """Alpha-Beta 剪枝版 Minimax，结果与朴素版完全相同。

    - A（MAX）节点：value = max(子节点)；alpha = max(alpha, value)；
      alpha >= beta 时剪枝（B 的祖先已有更小选择，不会走到这里）。
    - B（MIN）节点：value = min(子节点)；beta = min(beta, value)；
      alpha >= beta 时剪枝。
    """
    if stats is not None:
        stats.nodes += 1
    if state.depth >= horizon or not state.remaining:
        return state.score

    if state.turn == 'A':
        value = -INF
        for t in sorted(state.remaining):
            value = max(value, alpha_beta(apply_move(state, t, dist),
                                          horizon, dist, alpha, beta, stats))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = INF
        for t in sorted(state.remaining):
            value = min(value, alpha_beta(apply_move(state, t, dist),
                                          horizon, dist, alpha, beta, stats))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def alpha_beta_root(state: GameState, horizon: int, dist, stats: Optional[SearchStats] = None):
    """根节点版 Alpha-Beta：返回 (最优首步, 最优值, 各首步的精确博弈值)。

    注意：这里对每个首步都用**全新的 (α, β) = (−∞, +∞)** 求值，而不是在根节点
    复用递增的 α。原因：复用 α 会让后续首步的值被"截断"为不精确的上界
    （fail-soft 剪枝返回值），虽然不影响最优首步的判定，但 all_moves 明细会失真。
    本题要求输出各首步的博弈值用于分析，故牺牲少量剪枝换取明细数值的精确性。
    """
    stats0 = stats if stats is not None else SearchStats()
    stats0.nodes += 1
    best_target, best_value, all_moves = None, -INF, {}
    if state.depth >= horizon or not state.remaining:
        return None, state.score, {}
    for t in sorted(state.remaining):
        value = alpha_beta(apply_move(state, t, dist), horizon, dist, -INF, INF, stats0)
        all_moves[t] = value
        if value > best_value:
            best_value, best_target = value, t
    return best_target, best_value, all_moves


def best_first_move(horizon: int, dist, algorithm: str = 'alpha_beta'):
    """求 A 的最优首步与最优收益差。

    返回 (best_target, best_value, stats, all_moves)；
    all_moves 记录每个首步对应的博弈值，便于报告展示"H=2、H=4 的决策差异"。
    """
    root_search = alpha_beta_root if algorithm == 'alpha_beta' else minimax_root
    stats = SearchStats()
    initial = GameState(pos={'A': 'A', 'B': 'B'},
                        remaining=set(TREASURE_VALUES), turn='A', depth=0)
    best_target, best_value, all_moves = root_search(initial, horizon, dist, stats)
    return best_target, best_value, stats, all_moves


# ---------------- 主程序 ----------------

if __name__ == "__main__":
    names = ['A', 'B', 'T1', 'T2', 'T3', 'T4']

    # ---------- 1) 距离矩阵 ----------
    dist_list = distance_matrix()
    dist = {p: {q: dist_list[i][j] for j, q in enumerate(names)}
            for i, p in enumerate(names)}
    print("题目四 6×6 距离矩阵（行=起点，列=终点）")
    print("     " + "".join(f"{n:>5}" for n in names))
    for i, p in enumerate(names):
        print(f"{p:>4} " + "".join(f"{d:>5}" for d in dist_list[i]))

    # ---------- 2) 博弈搜索：H=2 / H=4，Minimax vs Alpha-Beta ----------
    expected = {2: ("T2", -1), 4: ("T3", 3)}
    print("\n搜索深度对比（Minimax vs Alpha-Beta）")
    print(f"{'H':>3} {'算法':>12} {'最优首步':>8} {'最优收益差':>10} {'展开节点':>8}")
    assert_log = []
    for H in (2, 4):
        for algo in ('minimax', 'alpha_beta'):
            bt, bv, stats, moves = best_first_move(H, dist, algo)
            print(f"{H:>3} {algo:>12} {bt:>8} {bv:>10} {stats.nodes:>8}")
            assert_log.append((H, algo, bt, bv, moves.get))
            if algo == 'minimax':
                assert (bt, bv) == expected[H], f"H={H} 预期 {expected[H]}，实际 {(bt, bv)}"

    # ---------- 3) 各首步的博弈值明细（报告素材） ----------
    print("\n各首步的博弈值（Alpha-Beta）：")
    for H in (2, 4):
        bt, bv, stats, moves = best_first_move(H, dist, 'alpha_beta')
        detail = "  ".join(f"{t}:{moves[t]:+d}" for t in sorted(moves))
        print(f"H={H}: {detail}   → 最优 {bt} ({bv:+d})")

    # ---------- 4) 预期输出（JSON 格式） ----------
    print("\n预期输出（PDF）：")
    for H in (2, 4):
        bt, bv, _, _ = best_first_move(H, dist, 'alpha_beta')
        print(f'H = {H}   =>  {{"best_target": "{bt}", "best_value": {bv}}}')

    assert best_first_move(2, dist, 'alpha_beta')[:2] == ("T2", -1)
    assert best_first_move(2, dist, 'minimax')[:2] == ("T2", -1)
    assert best_first_move(4, dist, 'alpha_beta')[:2] == ("T3", 3)
    assert best_first_move(4, dist, 'minimax')[:2] == ("T3", 3)
    print("\n全部断言通过 ✓")
