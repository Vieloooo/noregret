from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import auto, IntEnum
from functools import partial
import gzip
import hashlib
from importlib import import_module
from json import dump, dumps, load, loads
from math import inf
import os
from pathlib import Path
import pickle
from random import choices
import threading
from typing import Any

from ordered_set import OrderedSet
import numpy as np
import numpy.linalg as LA


def euclidean_projection_on_probability_simplex(input_):
    """Euclidean projection of the input on a probability simplex.

    >>> euclidean_projection_on_probability_simplex(np.array([0.2, 0.5, 0.3]))
    array([0.2, 0.5, 0.3])
    >>> euclidean_projection_on_probability_simplex(np.array([0.2, -0.3, 2]))
    array([0., 0., 1.])
    >>> euclidean_projection_on_probability_simplex(np.array([5, 5, 5, 5]))
    array([0.25, 0.25, 0.25, 0.25])
    >>> euclidean_projection_on_probability_simplex(np.array([10, 0, 0]))
    array([1., 0., 0.])
    >>> euclidean_projection_on_probability_simplex(np.array([0.6]))
    array([1.])
    >>> euclidean_projection_on_probability_simplex(np.array([0, 0, 0, 0, 0]))
    array([0.2, 0.2, 0.2, 0.2, 0.2])

    :param input_: The input to be projected.
    :return: The projection output.
    """
    sorted_input = np.flip(np.sort(input_))
    cumsum_sorted_input = sorted_input.cumsum()
    indices = np.arange(1, input_.size + 1)
    conditions = (sorted_input + (1 - cumsum_sorted_input) / indices) > 0
    rho = np.where(conditions)[0].max() + 1
    lambda_ = (1 - cumsum_sorted_input[rho - 1]) / rho
    output = (input_ + lambda_).clip(0)

    return output


def stationary_distribution(stochastic_matrix):
    P = stochastic_matrix
    eigenvalues, eigenvectors = LA.eig(P)
    pi = eigenvectors[:, np.isclose(eigenvalues, 1)][:, 0]
    pi /= pi.sum()
    pi = pi.real

    assert np.allclose(pi @ P, pi)

    return pi


def import_string(dotted_path):
    """Import an object from a module.

    >>> import_string('math.inf')
    inf

    :param dotted_path: The dotted path of the object to import.
    :return: Imported object.
    """
    module_path, class_name = dotted_path.rsplit('.', 1)

    return getattr(import_module(module_path), class_name)


def split(values, counts):
    """Split concatenated values.

    >>> split([0, 1, 2, 3, 4, 5], [3, 0, 1, 2])
    [[0, 1, 2], [], [3], [4, 5]]

    :param values: Values to be split.
    :param counts: The size of the partitions.
    :return: The split values.
    """
    splits = []
    begin = 0

    for count in counts:
        end = begin + count

        splits.append(values[begin:end])

        begin = end

    return splits


def sample(values, probabilities):
    return choices(values, probabilities)[0]


class Serializable(ABC):
    @classmethod
    @abstractmethod
    def deserialize(cls, raw_data):
        pass

    @classmethod
    def load(cls, *args, **kwargs):
        return cls.deserialize(load(*args, **kwargs))

    @classmethod
    def loads(cls, *args, **kwargs):
        return cls.deserialize(loads(*args, **kwargs))

    @abstractmethod
    def serialize(self):
        pass

    def dump(self, *args, **kwargs):
        return dump(self.serialize(), *args, **kwargs)

    def dumps(self, *args, **kwargs):
        return dumps(self.serialize(), *args, **kwargs)


@dataclass
class TreeFormSequentialDecisionProcess(Serializable):
    """Tree-Form Sequential Decision Process (TFSDP).

    这是一种把“带信息集的不完全信息/部分可观测顺序决策”抽象成树结构的表示。
    在 CFR（Counterfactual Regret Minimization）/序列形式（sequence form）里，
    我们经常需要：

    - 在**决策点**（decision point，类似信息集 $I$）上列出可选动作 $a \in A(I)$；
    - 在**观测点**（observation point）上把不可控的信号/观测 $s$ 汇总成分支；
    - 把“到达某个信息集并采取某动作”编码为一个**序列**（sequence）边 $(I,a)$，
        从而把策略表示成序列权重 $x(\sigma)$。

    本类的核心用途：
    1) 在树上做动态规划，计算最优回应（best response）；
    2) 给定一个行为策略（behavioral strategy）$\pi$，计算每个决策点的
            反事实/局部 Q 值（counterfactual utilities），供 CFR 更新遗憾值。
    """

    class NodeType(IntEnum):
        DECISION_POINT = auto()
        OBSERVATION_POINT = auto()
        END_OF_THE_DECISION_PROCESS = auto()

    @classmethod
    def deserialize_all(cls, raw_data):
        return list(map(cls.deserialize, raw_data))

    @classmethod
    def deserialize(cls, raw_data):
        transitions = {}
        node_types = {}

        for raw_transition in raw_data:
            parent_edge = tuple(raw_transition['parent_edge'])
            node_id = raw_transition['node']['id']
            node_type = getattr(cls.NodeType, raw_transition['node']['type'])
            transitions[parent_edge] = node_id
            node_types[node_id] = node_type

        return cls(transitions, node_types)

    transitions: Any
    """Entries are assumed be topologically sorted."""
    node_types: Any
    nodes: Any = field(init=False, default_factory=OrderedSet)
    decision_points: Any = field(init=False, default_factory=OrderedSet)
    observation_points: Any = field(init=False, default_factory=OrderedSet)
    sequences: Any = field(init=False, default_factory=OrderedSet)
    parent_sequences: Any = field(init=False, default_factory=dict)
    actions: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )
    signals: Any = field(
        init=False,
        default_factory=partial(defaultdict, OrderedSet),
    )

    def __post_init__(self):
        """在 TFSDP 原始转移 `transitions` 基础上派生出各种索引结构。

        这一步的作用类似：把一棵“历史-动作”树整理成 CFR/序列形式需要的数据结构。

        - `self.sequences`：收集所有“序列边”（root 也视为一个空序列 `()`）。
            对 CFR 来说，序列 $\sigma=(I,a)$ 表示“到达信息集 $I$ 后选择动作 $a$”。
        - `self.parent_sequences[p]`：每个节点对应的父序列（用于把行为策略转为序列策略）。
        - `self.actions[j]`：每个决策点/信息集 $j$ 的动作集合 $A(j)$。
        - `self.signals[p]`：观测点上的信号集合（不可控分支，类似公共/私有观测）。

        注意：这里的 parent_edge 形如 `(parent_node, event)`；
        对决策点来说 event 是动作；对观测点来说 event 是信号。
        """
        self.nodes.update(self.transitions.values())

        for parent_edge, p in self.transitions.items():
            match self.node_types[p]:
                case self.NodeType.DECISION_POINT:
                    self.decision_points.add(p)
                case self.NodeType.OBSERVATION_POINT:
                    self.observation_points.add(p)
                case self.NodeType.END_OF_THE_DECISION_PROCESS:
                    pass
                case _:
                    raise ValueError('unknown node type')

            if parent_edge:
                parent, event = parent_edge

                match self.node_types[parent]:
                    case self.NodeType.DECISION_POINT:
                        is_sequence = True
                        parent_sequence = parent_edge

                        self.actions[parent].add(event)
                    case self.NodeType.OBSERVATION_POINT:
                        is_sequence = False
                        parent_sequence = self.parent_sequences[parent]

                        self.signals[parent].add(event)
                    case self.NodeType.END_OF_THE_DECISION_PROCESS:
                        raise ValueError('parent is an end of the tfsdp')
                    case _:
                        raise ValueError('unknown parent node type')
            else:
                is_sequence = True
                parent_sequence = parent_edge

            if is_sequence:
                self.sequences.add(parent_edge)

            self.parent_sequences[p] = parent_sequence

    def behavioral_uniform_strategy(self):
        strategy = {
            j: np.full(len(self.actions[j]), 1 / len(self.actions[j]))
            for j in self.decision_points
        }

        return strategy

    def behavioral_best_response(self, utility):
        """给定“边上的即时效用” `utility`，计算行为策略意义下的最优回应。

        这在 CFR 里对应：
        - 在固定其他玩家策略与机会分布后，求某个玩家的 best response（用于 exploitability
            或者某些变体里的外层循环）。

        这里 `utility[k]` 表示序列边（self.sequences 的第 k 个）对应的即时收益
        $u(\sigma)$。对一个决策点（信息集）$I$，最优回应满足 Bellman 形式：

        $$
        V(I) = \max_{a\in A(I)} \bigl(u(I,a) + V(\text{next}(I,a))\bigr).
        $$

        对观测点 $O$（不可控信号分支），这里采用“把所有信号分支相加”的结构：

        $$
        V(O) = \sum_{s\in S(O)} V(\text{next}(O,s)).
        $$

        这等价于把观测点看作“确定会展开的并行分支/信息集切分”；在很多实现中，
        机会节点会用期望 $\sum_s p(s)V(\cdot)$，而此 TFSDP 把概率通常吸收到 `utility`
        或图结构里，所以这里直接求和。

        :param utility: shape = (#sequences,) 的向量，按 `self.sequences` 索引。
        :return: (behavioral_strategy, root_value)
            - behavioral_strategy[j] 是在决策点 j 的 one-hot 最优动作。
            - root_value 是从根节点开始的最优价值。
        """
        strategy = {}  # 最优回应的行为策略：每个决策点 j -> 一个 one-hot 分布（选中的动作概率为 1）
        V = defaultdict(int)  # 价值函数 V[p]：从节点 p 往后、在“最优回应/确定性展开规则”下的累积价值

        for p in reversed(self.nodes):  # 自底向上（逆拓扑）做动态规划；保证 children 的 V 已经算好
            match self.node_types[p]:  # 根据节点类型分别处理
                case self.NodeType.DECISION_POINT:
                    V[p] = -inf  # 决策点取 max：先把当前最优值初始化成负无穷
                    index = None  # 记录使 V[p] 达到最大值的动作索引 i（对应 self.actions[p] 的枚举顺序）

                    for i, a in enumerate(self.actions[p]):  # 枚举该信息集/决策点 p 的每个动作 a
                        # 这行对应 Bellman 里的 Q 值：Q(p,a) = u(p,a) + V(next(p,a))
                        value = (
                            utility[self.sequences.index((p, a))]  # u(p,a)：该序列边 (p,a) 的即时收益
                            + V[self.transitions[p, a]]  # 续行价值：走到 child 节点后的最优价值 V[next]
                        )

                        if V[p] < value:  # 如果该动作更优，就更新最优值与 argmax
                            V[p] = value
                            index = i

                    strategy[p] = np.zeros(len(self.actions[p]))  # one-hot：默认都不选
                    strategy[p][index] = 1  # 把最优动作对应位置置 1（确定性最优回应）
                case self.NodeType.OBSERVATION_POINT:
                    # 观测点没有“我方决策”，表示观测/信号导致的信息集切分。
                    # 这里按 TFSDP 的约定将所有信号分支的价值相加：V(p)=sum_s V(next(p,s))
                    for s in self.signals[p]:  # 枚举所有可能信号 s
                        V[p] += V[self.transitions[p, s]]  # 把每个信号分支的续行价值累加

        return strategy, V[self.nodes[0]]  # 返回：各决策点的最优回应策略 + 根节点的最优总价值

    def sequence_form_best_response(self, utility):
        strategy, value = self.behavioral_best_response(utility)

        return self.behavioral_to_sequence_form(strategy), value

    def behavioral_to_sequence_form(self, behavioral_strategy):
        strategy = np.zeros(len(self.sequences))
        strategy[self.sequences.index(())] = 1

        for j in self.decision_points:
            p_j = self.parent_sequences[j]

            for i, a in enumerate(self.actions[j]):
                strategy[self.sequences.index((j, a))] = (
                    behavioral_strategy[j][i]
                    * strategy[self.sequences.index(p_j)]
                )

        return strategy

    def counterfactual_utilities(self, behavioral_strategy, utility):
        """计算每个决策点（信息集）的反事实（counterfactual）动作价值。

        这一步是 CFR 的“核心读数”之一。
        在标准两人零和 CFR 中，给定当前策略 $\pi$，对玩家 $i$ 的信息集 $I$、动作 $a$，
        定义反事实价值（常见记号）为：

        $$
        v_i^{\pi}(I,a) = \sum_{h\in I} \pi_{-i}(h)\,\pi_c(h)\, u_i^{\pi}(h\cdot a),
        $$

        以及信息集价值

        $$
        v_i^{\pi}(I) = \sum_{a\in A(I)} \pi_i(a\mid I)\, v_i^{\pi}(I,a).
        $$

        这里的实现把“对手与机会到达权重”折叠进 TFSDP 的结构/`utility` 中，
        因而我们在树上做一次自底向上的动态规划即可得到类似的局部 Q 值：

        - 对决策点 $I$：
            $$
            V(I) = \sum_{a\in A(I)} \pi(a\mid I)\,[u(I,a)+V(\text{next}(I,a))].
            $$
        - 对观测点 $O$：
            $$
            V(O) = \sum_{s\in S(O)} V(\text{next}(O,s)).
            $$

        然后对每个决策点输出动作级别的“反事实效用/局部 Q 值”：

        $$
        Q(I,a) = u(I,a) + V(\text{next}(I,a)).
        $$

        在 CFR 更新里，立即遗憾（instant regret）通常是：

        $$
        r(I,a) = Q(I,a) - V(I).
        $$

        本函数返回的是 $Q(I,a)$（命名为 utilities[j][i]），以及未显式返回但可由
        上式组合得到的 $V(I)$。
        """
        V = defaultdict(int)  # V[p]：在给定行为策略 π 下，从节点 p 往后的“期望/汇总”价值（用于计算 Q）

        for p in reversed(self.nodes):  # 同样自底向上；先算出每个节点的 V[p]
            match self.node_types[p]:  # 决策点：按 π 加权求期望；观测点：按分支求和
                case self.NodeType.DECISION_POINT:
                    for i, a in enumerate(self.actions[p]):  # 遍历信息集/决策点 p 的每个动作 a
                        # 期望形式 Bellman：V(p)=sum_a π(a|p) * (u(p,a)+V(next(p,a)))
                        V[p] += (
                            behavioral_strategy[p][i]  # π(a|p)：在 p 处选择动作 a 的概率
                            * (
                                utility[self.sequences.index((p, a))]  # u(p,a)：序列边 (p,a) 的即时收益
                                + V[self.transitions[p, a]]  # V(next)：采取 a 后到子节点的续行价值
                            )
                        )
                case self.NodeType.OBSERVATION_POINT:
                    # 观测点仍是“展开所有信号分支并汇总”的约定：V(p)=sum_s V(next(p,s))
                    for s in self.signals[p]:  # 枚举信号 s
                        V[p] += V[self.transitions[p, s]]  # 累加每个信号分支的价值

        utilities = {}  # 输出：每个决策点 j -> 每个动作的 Q(j,a)（反事实/局部动作价值）

        for j in self.decision_points:  # 逐个决策点（信息集）构造动作价值向量
            utilities[j] = np.empty(len(self.actions[j]))  # utilities[j][i] 对应第 i 个动作的 Q 值

            for i, a in enumerate(self.actions[j]):  # 枚举动作 a
                # 这里返回 Q(j,a)=u(j,a)+V(next(j,a))；CFR 里常用它与 V(j) 形成遗憾 r(j,a)
                utilities[j][i] = (
                    utility[self.sequences.index((j, a))]  # u(j,a)
                    + V[self.transitions[j, a]]  # V(next)
                )

        return utilities  # 返回每个信息集的动作价值（用于 r(j,a)=Q(j,a)-V(j) 等更新）

    def serialize(self):
        raw_data = []

        for parent_edge, p in self.transitions.items():
            raw_data.append(
                {
                    'parent_edge': parent_edge,
                    'node': {'id': p, 'type': self.node_types[p].name},
                },
            )

        return raw_data

    # Backward-compatible alias: older code (e.g. games.py) expects `to_list()`.
    def to_list(self):
        return self.serialize()


