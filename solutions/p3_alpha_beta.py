# -*- coding: utf-8 -*-
"""题目三：极大极小博弈树的最佳得分（Alpha-Beta 剪枝）

在一棵极大极小博弈树上，根节点由 MAX 玩家执棋：MAX 最大化得分，MIN 最小化得分。
请通过 Alpha-Beta 剪枝求对 MAX 玩家而言的最佳得分。

规则
----
- 树的结构已定义：非叶子节点有 children 属性，叶子节点有 value 属性。
- 根节点是 MAX 玩家；子节点从左到右依次评估。

数据结构定义（Python 示例）
---------------------------
class TreeNode:
    def __init__(self, value=None, children=None):
        self.value = value       # 叶子节点的得分
        self.children = children if children is not None else []  # 子节点列表

输入 / 输出
-----------
- 输入：树的根节点（PDF 另附 JSON 输入格式样例，等价实现见 alpha_beta_json）。
- 输出：int，对 MAX 玩家而言的最佳得分。

参考：实验二要求_new.pdf 第 8-13 页
"""

import random
from typing import Any, Dict, List, Optional

INF = float("inf")


class TreeNode:
    """博弈树节点：叶子节点有 value，非叶子节点有 children。"""

    def __init__(self, value=None, children=None):
        self.value = value  # 叶子节点的得分
        self.children = children if children is not None else []  # 子节点列表


class Stats:
    """统计叶子节点的访问次数，用于对照实验（Minimax vs Alpha-Beta、不同访问顺序）。"""

    def __init__(self):
        self.leaf_visits = 0


def is_leaf(node: TreeNode) -> bool:
    return not node.children


def minimax(node: TreeNode, maximizing: bool = True, stats: Optional[Stats] = None) -> int:
    """朴素 Minimax：不剪枝，访问全部叶子节点。"""
    if is_leaf(node):
        if stats is not None:
            stats.leaf_visits += 1
        return node.value
    values = [minimax(child, not maximizing, stats) for child in node.children]
    return max(values) if maximizing else min(values)


def alpha_beta(
    node: TreeNode,
    maximizing: bool = True,
    alpha: float = -INF,
    beta: float = INF,
    stats: Optional[Stats] = None,
) -> int:
    """Alpha-Beta 剪枝。

    - alpha：MAX 节点当前能保证的得分下界（"我方至少能拿多少"）；
    - beta ：MIN 节点当前能容忍的得分上界（"对手最多让我拿多少"）；
    - 当 alpha >= beta 时，当前节点的取值无论如何都不可能影响祖先的决策，
      剩余子树可以直接剪掉，结果与朴素 Minimax 完全相同。
    """
    if is_leaf(node):
        if stats is not None:
            stats.leaf_visits += 1
        return node.value

    if maximizing:
        value = -INF
        for child in node.children:
            value = max(value, alpha_beta(child, False, alpha, beta, stats))
            alpha = max(alpha, value)
            if alpha >= beta:  # β 剪枝：MIN 祖先已有更好的选择
                break
        return value
    else:
        value = INF
        for child in node.children:
            value = min(value, alpha_beta(child, True, alpha, beta, stats))
            beta = min(beta, value)
            if alpha >= beta:  # α 剪枝：MAX 祖先已有更好的选择
                break
        return value


def alpha_beta_json(
    node: Dict[str, Any],
    alpha: float = -INF,
    beta: float = INF,
    stats: Optional[Stats] = None,
) -> int:
    """JSON 输入格式（含 is_maximizing_player 标志）的等价实现。

    与 alpha_beta 的"逐层交替"约定不同，这里以每个节点自带的
    is_maximizing_player 标志为准，对任意深度的 JSON 树都有效。
    """
    if "value" in node:
        if stats is not None:
            stats.leaf_visits += 1
        return node["value"]

    maximizing = node.get("is_maximizing_player", True)
    value = -INF if maximizing else INF
    for child in node["children"]:
        result = alpha_beta_json(child, alpha, beta, stats)
        if maximizing:
            value = max(value, result)
            alpha = max(alpha, value)
        else:
            value = min(value, result)
            beta = min(beta, value)
        if alpha >= beta:
            break
    return value


def build_sample_tree() -> TreeNode:
    """按 PDF 第 12 页构建示例（注意：max2 被 min1 与 min2 共享，形如 DAG）。"""
    leaf1, leaf2, leaf3 = TreeNode(3), TreeNode(5), TreeNode(6)
    leaf4, leaf5, leaf6 = TreeNode(9), TreeNode(1), TreeNode(2)
    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])
    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])  # 共享 max2
    return TreeNode(children=[min1, min2])


def build_distinct_tree() -> TreeNode:
    """与 PDF 树形相同的严格树（无共享子树），供开销对比实验使用。"""

    def leafs(*vs):
        return [TreeNode(v) for v in vs]

    max1 = TreeNode(children=leafs(3, 5))
    max2a = TreeNode(children=leafs(6, 9))
    max2b = TreeNode(children=leafs(6, 9))
    max3 = TreeNode(children=leafs(1, 2))
    min1 = TreeNode(children=[max1, max2a])
    min2 = TreeNode(children=[max2b, max3])
    return TreeNode(children=[min1, min2])


def random_tree(depth: int, b: int = 3) -> TreeNode:
    """随机博弈树：深度 depth，每个非叶子节点 b 个子节点，叶子随机取值 0-99。"""
    if depth == 0:
        return TreeNode(random.randint(0, 99))
    return TreeNode(children=[random_tree(depth - 1, b) for _ in range(b)])


if __name__ == "__main__":
    # ---------- 1) PDF 测试用例 ----------
    root = build_sample_tree()
    s_mm = Stats()
    print("题目三 测试用例（PDF 预期输出 5）")
    print("Minimax 结果:", minimax(root, True, s_mm), "，叶子访问:", s_mm.leaf_visits)
    s_ab = Stats()
    result = alpha_beta(root, True, -INF, INF, s_ab)
    print("Alpha-Beta 结果:", result, "，叶子访问:", s_ab.leaf_visits,
          "（与 PDF 预期 5 一致:", result == 5, "）")
    assert result == 5

    # ---------- 2) JSON 输入格式（PDF 第 10-11 页） ----------
    json_tree = {
        "tree_structure": {
            "is_maximizing_player": True,
            "children": [
                {"is_maximizing_player": False, "children": [
                    {"is_maximizing_player": True, "children": [{"value": 3}, {"value": 5}]},
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                ]},
                {"is_maximizing_player": False, "children": [
                    {"is_maximizing_player": True, "children": [{"value": 6}, {"value": 9}]},
                    {"is_maximizing_player": True, "children": [{"value": 1}, {"value": 2}]},
                ]},
            ],
        }
    }
    print("\nJSON 输入格式结果:", alpha_beta_json(json_tree["tree_structure"]))

    # ---------- 3) 对照实验：Minimax vs Alpha-Beta（严格树）与不同访问顺序 ----------
    tree = build_distinct_tree()
    s = Stats(); minimax(tree, True, s)
    print("\n严格树 Minimax 叶子访问:", s.leaf_visits)
    s = Stats(); alpha_beta(tree, True, -INF, INF, s)
    print("严格树 Alpha-Beta（左→右）叶子访问:", s.leaf_visits)

    reversed_tree = build_distinct_tree()

    def reverse_children(node: TreeNode) -> None:
        node.children = list(reversed(node.children))
        for ch in node.children:
            reverse_children(ch)

    reverse_children(reversed_tree)
    s = Stats()
    print("Alpha-Beta（右→左）结果:", alpha_beta(reversed_tree, True, -INF, INF, s),
          "，叶子访问:", s.leaf_visits)

    # ---------- 4) 随机树上的开销对比（报告素材） ----------
    print("\n随机树开销对比（深度 4、每节点 3 子节点、5 次取平均）:")
    mm_total = ab_total = 0
    for _ in range(5):
        t = random_tree(4)
        s = Stats(); minimax(t, True, s); mm_total += s.leaf_visits
        s = Stats(); alpha_beta(t, True, -INF, INF, s); ab_total += s.leaf_visits
    print(f"Minimax 平均叶子访问 {mm_total // 5}，Alpha-Beta 平均 {ab_total // 5}")

    print("\n全部断言通过 ✓")
