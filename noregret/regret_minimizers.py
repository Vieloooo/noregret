from abc import ABC, abstractmethod
from dataclasses import dataclass, field, KW_ONLY
from functools import partial
from itertools import repeat
from math import inf, isinf
from typing import Any

from scipy.special import softmax
import numpy as np

from noregret.utilities import (
    euclidean_projection_on_probability_simplex,
    sample,
    split,
    stationary_distribution,
)


@dataclass
class RegretMinimizer(ABC):
    """Regret minimizer."""

    dimension: Any
    _: KW_ONLY
    gamma: Any = 0
    is_time_symmetric: bool = False
    iteration_count: Any = 0
    strategies: Any = field(default_factory=list)
    previous_strategy: Any = None
    weight_sum: Any = 0
    average_strategy: Any = None
    utilities: Any = field(default_factory=list)
    previous_utility: Any = None
    cumulative_utility: Any = None

    def __post_init__(self):
        if self.previous_strategy is None:
            self.previous_strategy = np.full(
                self.dimension,
                1 / self.dimension,
            )

        if self.average_strategy is None:
            self.average_strategy = np.full(self.dimension, 1 / self.dimension)

        if self.previous_utility is None:
            self.previous_utility = np.zeros(self.dimension)

        if self.cumulative_utility is None:
            self.cumulative_utility = np.zeros(self.dimension)

    @abstractmethod
    def next_strategy(self, prediction=False):
        pass

    def undo_next_strategy(self):
        self.strategies.pop()

    def observe_utility(self, utility):
        if len(self.strategies) == len(self.utilities):
            raise ValueError('next strategy not yet outputted')
        elif len(self.strategies) != len(self.utilities) + 1:
            raise ValueError('more than one strategies were outputted')

        strategy = self.strategies[-1]
        self.iteration_count += 1

        if self.is_time_symmetric:
            self.strategies.clear()

        self.previous_strategy = strategy.copy()

        self.update_average_strategy()

        if not self.is_time_symmetric:
            self.utilities.append(utility.copy())

        self.previous_utility = utility.copy()
        self.cumulative_utility += utility

    def update_average_strategy(self):
        if self.gamma > 0 and isinf(self.gamma):
            weight = inf
        else:
            weight = self.iteration_count ** self.gamma

        self.weight_sum += weight

        if isinf(weight):
            self.average_strategy = self.previous_strategy
        else:
            self.average_strategy += (
                weight
                * (self.previous_strategy - self.average_strategy)
                / self.weight_sum
            )


@dataclass
class ProbabilitySimplexRegretMinimizer(RegretMinimizer, ABC):
    """Regret minimizer.

    Assumes a linear utility function, and optimizes over the
    probability simplex.
    """

    _: KW_ONLY
    regrets: Any = field(default_factory=list)
    previous_regrets: Any = None
    cumulative_regrets: Any = None
    cumulative_regret: Any = 0

    def __post_init__(self):
        super().__post_init__()

        if self.previous_regrets is None:
            self.previous_regrets = np.zeros(self.dimension)

        if self.cumulative_regrets is None:
            self.cumulative_regrets = np.zeros(self.dimension)

    def observe_utility(self, utility):
        super().observe_utility(utility)

        regrets = utility - (self.previous_strategy @ utility)

        if not self.is_time_symmetric:
            self.regrets.append(regrets.copy())

        self.previous_regrets = regrets.copy()
        self.cumulative_regrets += regrets
        self.cumulative_regret = self.cumulative_regrets.max()


@dataclass
class FollowTheRegularizedLeader(ProbabilitySimplexRegretMinimizer, ABC):
    """Follow the regularized leader (FTLR)."""

    learning_rate: Any


@dataclass
class MultiplicativeWeightsUpdate(FollowTheRegularizedLeader):
    """Multiplicative weights update (MWU)."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_utility
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = prediction + self.cumulative_utility

        strategy = softmax(self.learning_rate * theta)

        self.strategies.append(strategy)

        return strategy


@dataclass
class EuclideanRegularization(FollowTheRegularizedLeader):
    """Euclidean regularization."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_utility
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = prediction + self.cumulative_utility

        strategy = euclidean_projection_on_probability_simplex(
            self.learning_rate * theta,
        )

        self.strategies.append(strategy)

        return strategy


@dataclass
class MirrorDescent(ProbabilitySimplexRegretMinimizer, ABC):
    """(Online) mirror descent (MR)."""

    learning_rate: Any


@dataclass
class OnlineGradientDescent(MirrorDescent):
    """Online gradient descent."""

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            theta = self.previous_utility
        else:
            theta = prediction

        strategy = euclidean_projection_on_probability_simplex(
            self.previous_strategy + self.learning_rate * theta,
        )

        self.strategies.append(strategy)

        return strategy


@dataclass
class RegretMatching(ProbabilitySimplexRegretMinimizer):
    """Regret matching."""

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_regrets
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.cumulative_regrets
            )

        return self._next_strategy(theta.clip(0))

    def _next_strategy(self, unnormalized_strategy):
        if np.allclose(unnormalized_strategy, 0):
            strategy = np.full(self.dimension, 1 / self.dimension)
        else:
            strategy = unnormalized_strategy / unnormalized_strategy.sum()

        self.strategies.append(strategy)

        return strategy


@dataclass
class RegretMatchingPlus(RegretMatching):
    """Regret matching plus."""

    _: KW_ONLY
    cumulative_regrets_plus: Any = None

    def __post_init__(self):
        super().__post_init__()

        if self.cumulative_regrets_plus is None:
            self.cumulative_regrets_plus = np.zeros(self.dimension)

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.cumulative_regrets_plus
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.cumulative_regrets_plus
            ).clip(0)

        return self._next_strategy(theta)

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.cumulative_regrets_plus += self.previous_regrets
        self.cumulative_regrets_plus = self.cumulative_regrets_plus.clip(0)


class DiscountedStrategyAveragingMixin:
    gamma: Any
    iteration_count: Any
    previous_strategy: Any
    weight_sum: Any
    average_strategy: Any

    def update_average_strategy(self):
        T = self.iteration_count
        weight = (T / (T + 1)) ** self.gamma
        self.weight_sum += weight
        self.average_strategy += (
            weight
            * (self.previous_strategy - self.average_strategy)
            / self.weight_sum
        )


@dataclass
class DiscountedRegretMatching(
        DiscountedStrategyAveragingMixin,
        RegretMatching,
):
    """Discounted regret matching (used in DCFR)."""

    _: KW_ONLY
    alpha: Any = 0
    beta: Any = 0
    discounted_regrets: Any = None

    def __post_init__(self):
        super().__post_init__()

        if self.discounted_regrets is None:
            self.discounted_regrets = np.zeros(self.dimension)

    def next_strategy(self, prediction=False):
        if prediction is False:
            theta = self.discounted_regrets
        else:
            if prediction is True:
                prediction = self.previous_utility

            theta = (
                prediction
                - prediction @ self.previous_strategy
                + self.discounted_regrets
            )
            T = self.iteration_count + 1
            theta[theta > 0] *= T ** self.alpha / (T ** self.alpha + 1)
            theta[theta < 0] *= T ** self.beta / (T ** self.beta + 1)

        return self._next_strategy(theta.clip(0))

    def observe_utility(self, utility):
        super().observe_utility(utility)

        self.discounted_regrets += self.previous_regrets
        T = self.iteration_count
        self.discounted_regrets[self.discounted_regrets > 0] *= (
            T ** self.alpha / (T ** self.alpha + 1)
        )
        self.discounted_regrets[self.discounted_regrets < 0] *= (
            T ** self.beta / (T ** self.beta + 1)
        )


@dataclass
class SwapRegretMinimizer(RegretMinimizer, ABC):
    """Swap regret minimizer."""


@dataclass
class ProbabilitySimplexSwapRegretMinimizer(SwapRegretMinimizer, ABC):
    """Swap regret minimizer.

    Optimizes over the probability simplex.
    """


@dataclass
class BlumMansour(ProbabilitySimplexSwapRegretMinimizer):
    """Blum-Mansour algorithm.

    External regret minimizers optimize over the probability simplex.
    """

    regret_minimizer_factory: Any = None
    _: KW_ONLY
    outputs: Any = field(init=False)
    external_regret_minimizers: Any = None

    def __post_init__(self):
        super().__post_init__()

        d = self.dimension
        self.outputs = np.full((d, d), 1 / d)

        if self.external_regret_minimizers is None:
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            self.external_regret_minimizers = tuple(
                map(self.regret_minimizer_factory, repeat(d, d)),
            )

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            for a, R in enumerate(self.external_regret_minimizers):
                self.outputs[:, a] = R.next_strategy(prediction)
        else:
            for a, R in enumerate(self.external_regret_minimizers):
                self.outputs[:, a] = R.next_strategy(
                    self.previous_strategy[a] * prediction,
                )

        strategy = stationary_distribution(self.outputs)

        self.strategies.append(strategy)

        return strategy

    def undo_next_strategy(self):
        raise NotImplementedError

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for a, R in enumerate(self.external_regret_minimizers):
            R.observe_utility(self.previous_strategy[a] * utility)


@dataclass
class SequenceFormPolytopeRegretMinimizer(RegretMinimizer, ABC):
    """Regret minimizer.

    Optimizes over the sequence-form polytope.
    """


@dataclass
class CounterfactualRegretMinimization(SequenceFormPolytopeRegretMinimizer):
    """Counterfactual regret minimization (CFR).

    Local regret minimizers optimize over the probability simplex.
    """

    tree_form_sequential_decision_process: Any
    regret_minimizer_factory: Any = RegretMatching
    _: KW_ONLY
    dimension: Any = field(init=False)
    behavioral_strategy: Any = field(init=False)
    previous_behavioral_strategy: Any = None
    local_regret_minimizers: Any = None

    def __post_init__(self):
        tfsdp = self.tree_form_sequential_decision_process  # 该玩家的 TFSDP（信息集/动作/序列边的树表示）
        self.dimension = len(tfsdp.sequences)  # 序列形式（sequence-form）策略向量维度 = 序列边数量（含 root 空序列）

        super().__post_init__()  # 初始化基类的序列形式 regret minimizer 记录（平均策略、累计效用等）

        self.behavioral_strategy = tfsdp.behavioral_uniform_strategy()  # 初始化为行为策略的均匀分布：每个信息集动作等概率

        if self.previous_behavioral_strategy is None:  # 保存“上一轮”的行为策略（用于 optimistic/prediction 变体）
            self.previous_behavioral_strategy = self.behavioral_strategy.copy()

        if self.local_regret_minimizers is None:  # 为每个信息集/决策点创建一个“局部”regret minimizer
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            # CFR 的关键：把大博弈分解成许多局部 simplex 优化问题。
            # 对每个决策点 j，局部算法只需要处理 |A(j)| 维的动作分布。
            self.local_regret_minimizers = {
                j: self.regret_minimizer_factory(len(tfsdp.actions[j]))  # 例如 RegretMatching(|A(j)|)
                for j in tfsdp.decision_points
            }

    def next_strategy(self, prediction=False):
        self.behavioral_strategy = {}  # 本轮输出的行为策略：j -> 概率向量 π(·|j)

        if prediction is False or prediction is True:
            # 常规路径：每个信息集的局部算法直接产出 π(·|j)
            # prediction=False：不使用预测；prediction=True：使用局部算法自身的默认预测（通常是 previous_utility）
            for j, R in self.local_regret_minimizers.items():
                self.behavioral_strategy[j] = R.next_strategy(prediction)
        else:
            # optimistic / predictive CFR 路径：prediction 不是 bool，而是一条“序列边效用”向量
            # 我们需要把它从 sequence-form 的边效用，映射成每个信息集动作的 Q 预测值。
            counterfactual_predictions = (
                self
                .tree_form_sequential_decision_process
                .counterfactual_utilities(
                    self.previous_behavioral_strategy,  # 用上一轮行为策略 π^{t-1}
                    prediction,  # 预测的边效用 û(σ)
                )
            )  # 得到：对每个信息集 j，预测的 Q̂(j,·)

            for j, R in self.local_regret_minimizers.items():
                # 把该信息集的动作级预测 Q̂(j,·) 交给局部算法，产生下一步策略
                self.behavioral_strategy[j] = R.next_strategy(
                    counterfactual_predictions[j],
                )

        # 将行为策略 π(·|j) 转为序列形式 realization plan x(σ)
        # x(())=1，且对每条序列边 (j,a)：x(j,a)=π(a|j)*x(parent_seq(j))
        strategy = (
            self
            .tree_form_sequential_decision_process
            .behavioral_to_sequence_form(self.behavioral_strategy)
        )

        self.strategies.append(strategy)  # 记录本轮输出的序列形式策略（供平均/日志等用途）

        return strategy  # 返回给上层（例如对偶更新或博弈求解外层）

    def undo_next_strategy(self):
        raise NotImplementedError

    def observe_utility(self, utility):
        super().observe_utility(utility)  # 记录本轮收到的“序列边效用”向量 u(σ)

        # 关键：把全局的边效用 u(σ) + 当前行为策略 π，变成每个信息集的动作价值 Q(j,·)
        # TFSDP.counterfactual_utilities 计算：Q(j,a)=u(j,a)+V(next(j,a))，其中 V 按 π 自底向上汇总
        counterfactual_utilities = (
            self
            .tree_form_sequential_decision_process
            .counterfactual_utilities(self.behavioral_strategy, utility)
        )

        for j, R in self.local_regret_minimizers.items():
            # 将信息集 j 的动作价值向量喂给局部 regret minimizer
            # 局部算法内部会形成即时遗憾 r(j,a)=Q(j,a)-E_{a'~π}[Q(j,a')] 并累积
            R.observe_utility(counterfactual_utilities[j])

        self.previous_behavioral_strategy = self.behavioral_strategy.copy()  # 为下一轮 prediction 路径缓存 π^{t}


@dataclass
class CounterfactualRegretMinimizationPlus(CounterfactualRegretMinimization):
    """Counterfactual regret minimization plus (CFR+)."""

    regret_minimizer_factory: Any = RegretMatchingPlus
    _: KW_ONLY
    gamma: Any = 1


@dataclass
class DiscountedCounterfactualRegretMinimization(
        DiscountedStrategyAveragingMixin,
        CounterfactualRegretMinimization,
):
    """Discounted counterfactual regret minimization (DCFR)."""

    regret_minimizer_factory: Any = None
    _: KW_ONLY
    alpha: Any = 1.5
    beta: Any = 0
    gamma: Any = 2

    def __post_init__(self):
        if self.regret_minimizer_factory is None:
            # DCFR：把“折扣”逻辑注入到局部 regret matching（对正/负遗憾分别折扣）
            # 这里用 partial 固定参数，仍保持“每个信息集一个局部算法”的 CFR 结构。
            self.regret_minimizer_factory = partial(
                DiscountedRegretMatching,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

        super().__post_init__()  # 继续复用 CFR 的初始化流程（创建 local_regret_minimizers 等）


@dataclass
class RegretCircuit(RegretMinimizer):
    """Regret circuit.

    Assumes a linear utility function.
    """


@dataclass
class CartesianProductRegretCircuit(RegretCircuit):
    """Regret circuit for the cartesian product operation.

    Optimizes over the cartesian product of sets, over which the components
    optimize.
    """

    component_regret_minimizers: Any
    _: KW_ONLY
    dimension: Any = field(init=False)

    def __post_init__(self):
        self.dimension = sum(self.dimensions)

        super().__post_init__()

    @property
    def dimensions(self):
        return np.array(
            [R.dimension for R in self.component_regret_minimizers],
        )

    def next_strategy(self, prediction=False):
        if prediction is False or prediction is True:
            predictions = repeat(prediction)
        else:
            predictions = split(prediction, self.dimensions)

        strategy = []

        for R, m in zip(self.component_regret_minimizers, predictions):
            strategy.append(R.next_strategy(m))

        strategy = np.concatenate(strategy)

        self.strategies.append(strategy)

        return strategy

    def undo_next_strategy(self):
        raise NotImplementedError

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for R, utility in zip(
                self.component_regret_minimizers,
                split(utility, self.dimensions),
        ):
            R.observe_utility(utility)


@dataclass
class ConvexHullRegretCircuit(RegretCircuit):
    """Regret circuit for the convex hull operation.

    Optimizes over the convex hull of sets, over which the components
    optimize.
    """

    component_regret_minimizers: Any
    regret_minimizer_factory: Any = None
    _: KW_ONLY
    dimension: Any = field(init=False)
    outputs: Any = field(init=False)
    previous_outputs: Any = None
    mixing_regret_minimizer: Any = None

    def __post_init__(self):
        if self.component_regret_minimizers:
            self.dimension = self.component_regret_minimizers[0].dimension
        else:
            self.dimension = 0

        super().__post_init__()

        self.outputs = np.full(
            (len(self.component_regret_minimizers), self.dimension),
            1 / self.dimension,
        )

        if self.previous_outputs is None:
            self.previous_outputs = self.outputs.copy()

        if self.mixing_regret_minimizer is None:
            if self.regret_minimizer_factory is None:
                raise ValueError('regret minimizer factory not set')

            self.mixing_regret_minimizer = self.regret_minimizer_factory(
                len(self.component_regret_minimizers),
            )

    def next_strategy(self, prediction=False):
        for i, R in enumerate(self.component_regret_minimizers):
            self.outputs[i] = R.next_strategy(prediction)

        if prediction is False or prediction is True:
            m = prediction
        else:
            m = self.previous_outputs @ prediction

        probabilities = self.mixing_regret_minimizer.next_strategy(m)
        strategy = self.outputs.T @ probabilities

        self.strategies.append(strategy)

        return strategy

    def undo_next_strategy(self):
        raise NotImplementedError

    def observe_utility(self, utility):
        super().observe_utility(utility)

        for R in self.component_regret_minimizers:
            R.observe_utility(utility)

        self.previous_outputs = self.outputs.copy()

        self.mixing_regret_minimizer.observe_utility(self.outputs @ utility)


@dataclass
class StochasticRegretMinimization(ABC):
    """Stochastic regret minimization."""

    extensive_form_game: Any

    @property
    def average_strategy_profile(self):
        return lambda state: (
            self._local_regret_minimizer(state).average_strategy
        )

    @abstractmethod
    def _local_regret_minimizer(self, state):
        pass

    def external_sampling(self):
        for player in self.extensive_form_game.players:
            self._external_sampling(
                player,
                self.extensive_form_game.initial_state,
            )

    def _external_sampling(self, player, state):
        if state.is_terminal():
            utility = state.utility(player)
        elif state.is_chance():
            actions, probabilities = zip(*state.chance_action_probabilities)
            action = sample(actions, probabilities)
            utility = self._external_sampling(player, state.apply(action))
        else:
            local_regret_minimizer = self._local_regret_minimizer(state)
            actions = state.actions
            probabilities = local_regret_minimizer.next_strategy()

            if state.player == player:
                utilities = list(
                    map(
                        partial(self._external_sampling, player),
                        map(state.apply, actions),
                    ),
                )
                utility = utilities @ probabilities

                local_regret_minimizer.observe_utility(utilities)
            else:
                action = sample(actions, probabilities)
                utility = self._external_sampling(player, state.apply(action))

                local_regret_minimizer.undo_next_strategy()

        return utility

    def outcome_sampling(self, reference_strategy_profile):
        for player in self.extensive_form_game.players:
            self._outcome_sampling(
                reference_strategy_profile,
                player,
                self.extensive_form_game.initial_state,
                1,
            )

    def _outcome_sampling(
            self,
            reference_strategy_profile,
            player,
            state,
            reference_reach_probability,
    ):
        if state.is_terminal():
            utility = state.utility(player) / reference_reach_probability
        elif state.is_chance():
            actions, probabilities = zip(*state.chance_action_probabilities)
            action = sample(actions, probabilities)
            utility = self._outcome_sampling(
                reference_strategy_profile,
                player,
                state.apply(action),
                reference_reach_probability,
            )
        else:
            local_regret_minimizer = self._local_regret_minimizer(state)
            actions = state.actions

            if state.player == player:
                probabilities = reference_strategy_profile(state)
                index = sample(range(len(actions)), probabilities)
                action = actions[index]
                probability = probabilities[index]
                utility = (
                    probability
                    * self._outcome_sampling(
                        reference_strategy_profile,
                        player,
                        state.apply(action),
                        probability * reference_reach_probability,
                    )
                )
                utilities = np.zeros(len(actions))
                utilities[index] = utility

                local_regret_minimizer.next_strategy()
                local_regret_minimizer.observe_utility(utilities)
            else:
                probabilities = local_regret_minimizer.next_strategy()
                action = sample(actions, probabilities)
                utility = self._outcome_sampling(
                    reference_strategy_profile,
                    player,
                    state.apply(action),
                    reference_reach_probability,
                )

                local_regret_minimizer.undo_next_strategy()

        return utility


@dataclass
class MonteCarloCounterfactualRegretMinimization(StochasticRegretMinimization):
    """Monte Carlo Counterfactual regret minimization (MCCFR)."""

    regret_minimizer_factory: Any = partial(
        RegretMatching,
        is_time_symmetric=True,
    )
    _: KW_ONLY
    local_regret_minimizers: Any = field(init=False, default_factory=dict)

    @property
    def iteration_count(self):
        iteration_count = 0

        for R in self.local_regret_minimizers.values():
            iteration_count += R.iteration_count

        return iteration_count

    def _local_regret_minimizer(self, state):
        if state.infoset in self.local_regret_minimizers:
            R = self.local_regret_minimizers[state.infoset]
        else:
            action_count = len(state.actions)
            R = self.regret_minimizer_factory(action_count)
            self.local_regret_minimizers[state.infoset] = R

        return R
