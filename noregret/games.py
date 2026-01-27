from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from functools import cache, partial
from itertools import permutations
from math import factorial
from typing import Any

from ordered_set import OrderedSet
from scipy.sparse import lil_array
import numpy as np

from noregret.utilities import (
    Serializable,
    split,
    TreeFormSequentialDecisionProcess,
)


@dataclass
class Game(ABC):
    """Game."""

    def __post_init__(self):
        self._verify()

    def _verify(self, **kwargs):
        pass

    @property
    @abstractmethod
    def player_count(self):
        pass

    @abstractmethod
    def dimension(self, player):
        pass

    @property
    def dimensions(self):
        return np.array(list(map(self.dimension, range(self.player_count))))

    @abstractmethod
    def utility(self, player, *opponent_strategies):
        pass

    @abstractmethod
    def value(self, player, *strategies):
        pass

    def values(self, *strategies):
        return np.array(
            [self.value(i, *strategies) for i in range(self.player_count)],
        )

    @abstractmethod
    def correlated_value(self, player, *strategies):
        pass

    def correlated_values(self, *strategies):
        return np.array(
            [
                self.correlated_value(i, *strategies)
                for i in range(self.player_count)
            ],
        )

    @abstractmethod
    def best_response(self, player, *opponent_strategies):
        pass

    def nash_gap(self, *strategies):
        gap = 0

        for i, value in enumerate(self.values(*strategies)):
            opponent_strategies = strategies[:i] + strategies[i + 1:]
            _, best_response_value = self.best_response(
                i,
                *opponent_strategies,
            )
            gap += best_response_value - value

        return gap

    def cce_gap(self, *strategies):
        average_strategies = list(map(partial(np.mean, axis=0), strategies))
        gap = 0

        for i, value in enumerate(self.correlated_values(*strategies)):
            average_opponent_strategies = (
                average_strategies[:i] + average_strategies[i + 1:]
            )
            _, best_response_value = self.best_response(
                i,
                *average_opponent_strategies,
            )
            gap += best_response_value - value

        return gap


@dataclass
class TwoPlayerGame(Game, ABC):
    """Two-player (2p) game.

    Row and column players are of indices 0 and 1, respectively.
    """

    def _verify(self, **kwargs):
        super()._verify(**kwargs)

        if self.player_count != 2:
            raise ValueError('number of players not 2')

    @property
    @abstractmethod
    def row_utilities(self):
        pass

    @property
    @abstractmethod
    def column_utilities(self):
        pass

    def dimension(self, player):
        match player:
            case 0:
                dimension = self.row_dimension
            case 1:
                dimension = self.column_dimension
            case _:
                raise ValueError(f'Player {player} does not exist')

        return dimension

    @property
    def row_dimension(self):
        return self.row_utilities.shape[0]

    @property
    def column_dimension(self):
        return self.row_utilities.shape[1]

    def utility(self, player, opponent_strategy):
        match player:
            case 0:
                utility = self.row_utility(opponent_strategy)
            case 1:
                utility = self.column_utility(opponent_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return utility

    def row_utility(self, column_strategy):
        return self.row_utilities @ column_strategy

    def column_utility(self, row_strategy):
        return row_strategy @ self.column_utilities

    def value(self, player, row_strategy, column_strategy):
        match player:
            case 0:
                value = self.row_value(row_strategy, column_strategy)
            case 1:
                value = self.column_value(row_strategy, column_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return value

    def row_value(self, row_strategy, column_strategy):
        return row_strategy @ self.row_utilities @ column_strategy

    def column_value(self, row_strategy, column_strategy):
        return row_strategy @ self.column_utilities @ column_strategy

    def correlated_value(self, player, row_strategies, column_strategies):
        match player:
            case 0:
                value = self.correlated_row_value(
                    row_strategies,
                    column_strategies,
                )
            case 1:
                value = self.correlated_column_value(
                    row_strategies,
                    column_strategies,
                )
            case _:
                raise ValueError(f'Player {player} does not exist')

        return value

    def correlated_row_value(self, row_strategies, column_strategies):
        return (
            row_strategies @ self.row_utilities * column_strategies
        ).sum(1).mean()

    def correlated_column_value(self, row_strategies, column_strategies):
        return (
            row_strategies @ self.column_utilities * column_strategies
        ).sum(1).mean()

    def best_response(self, player, opponent_strategy):
        match player:
            case 0:
                best_response = self.row_best_response(opponent_strategy)
            case 1:
                best_response = self.column_best_response(opponent_strategy)
            case _:
                raise ValueError(f'Player {player} does not exist')

        return best_response

    @abstractmethod
    def row_best_response(self, column_strategy):
        pass

    @abstractmethod
    def column_best_response(self, row_strategy):
        pass


@dataclass
class TwoPlayerZeroSumGame(TwoPlayerGame, ABC):
    """Two-player zero-sum (2p0s) game."""

    @property
    def column_utilities(self):
        return -self.row_utilities

    def values(self, row_strategy, column_strategy):
        value = self.row_value(row_strategy, column_strategy)

        return np.array((value, -value))

    def correlated_values(self, row_strategies, column_strategies):
        value = self.correlated_row_value(row_strategies, column_strategies)

        return np.array((value, -value))

    def nash_gap(self, row_strategy, column_strategy):
        _, row_best_response_value = self.row_best_response(column_strategy)
        _, column_best_response_value = self.column_best_response(row_strategy)

        return row_best_response_value + column_best_response_value

    def exploitability(self, row_strategy, column_strategy):
        return self.nash_gap(row_strategy, column_strategy) / 2


@dataclass
class NormalFormGame(Serializable, Game):
    """Normal-form game.

    Each player optimizes over the probability simplex.
    """

    @classmethod
    def deserialize(cls, raw_data):
        return cls(raw_data['actions'], np.array(raw_data['utilities']))

    actions: Any
    utilities: Any

    def __post_init__(self):
        super().__post_init__()

        self.actions = tuple(map(OrderedSet, self.actions))

    def _verify(self, *, utilities_shape=None, **kwargs):
        super()._verify(**kwargs)

        if utilities_shape is None:
            utilities_shape = (*map(len, self.actions), self.player_count)

        if self.utilities.shape != utilities_shape:
            raise ValueError('utilities do not match actions and players')

    @property
    def player_count(self):
        return len(self.actions)

    def utility(self, player, *opponent_strategies):
        raise NotImplementedError

    def value(self, player, *strategies):
        raise NotImplementedError

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError

    def serialize(self):
        return {'actions': self.actions, 'utilities': self.utilities.tolist()}


@dataclass
class TwoPlayerNormalFormGame(TwoPlayerGame, NormalFormGame):
    """Two-player (2p) normal-form game."""

    @property
    def row_actions(self):
        return self.actions[0]

    @property
    def column_actions(self):
        return self.actions[1]

    @property
    def row_utilities(self):
        return self.utilities[:, :, 0]

    @property
    def column_utilities(self):
        return self.utilities[:, :, 1]

    def row_best_response(self, column_strategy):
        strategy = np.zeros(len(self.row_actions))
        utility = self.row_utility(column_strategy)
        index = utility.argmax()
        strategy[index] = 1

        return strategy, utility[index]

    def column_best_response(self, row_strategy):
        strategy = np.zeros(len(self.column_actions))
        utility = self.column_utility(row_strategy)
        index = utility.argmax()
        strategy[index] = 1

        return strategy, utility[index]


@dataclass
class TwoPlayerZeroSumNormalFormGame(
        TwoPlayerZeroSumGame,
        TwoPlayerNormalFormGame,
):
    """Two-player zero-sum (2p0s) normal-form game.

    The utility matrix is from the viewpoint of the row player.
    """

    def _verify(self, **kwargs):
        super()._verify(
            **kwargs,
            utilities_shape=(len(self.row_actions), len(self.column_actions)),
        )

    @property
    def row_utilities(self):
        return self.utilities


@dataclass
class ExtensiveFormGame(Serializable, Game):
    """Extensive-form game (EFG).

    Each player optimizes over the sequence-form polytope.
    """

    @classmethod
    def deserialize(cls, raw_data):
        raise NotImplementedError

    # A list of TFSDP here, one for each player. 
    tree_form_sequential_decision_processes: Any
    utilities: Any

    @property
    def player_count(self):
        return len(self.tree_form_sequential_decision_processes)

    def utility(self, player, *opponent_strategies):
        raise NotImplementedError

    def value(self, player, *strategies):
        raise NotImplementedError

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError

    def serialize(self):
        raise NotImplementedError


@dataclass
class TwoPlayerExtensiveFormGame(TwoPlayerGame, ExtensiveFormGame):
    """Two-player (2p) extensive-form game (EFG)."""

    @classmethod
    def deserialize(cls, raw_data):
        tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        utilities = raw_data['utilities']

        # Unified sparse profile format: coords + per-player sparse value vectors.
        if isinstance(utilities, dict) and utilities.get('kind') == 'scipy.sparse.profile_per_player':
            if int(utilities.get('player_count', 2)) != 2:
                raise ValueError('utility is not of a 2-player game')
            if bool(utilities.get('zero_sum', False)):
                raise ValueError('expected general-sum utilities, got zero-sum')

            coords = np.asarray(utilities['coords'])
            payloads = list(utilities['values'])
            nnz = int(coords.shape[0])

            def _csr_vector_payload_to_dense(payload, length: int) -> np.ndarray:
                if payload.get('type') != 'csr':
                    raise ValueError('unsupported sparse utility type')
                shape = tuple(payload['shape'])
                if shape != (length, 1):
                    raise ValueError('unexpected sparse vector shape')
                indptr = payload['indptr']
                data = payload['data']
                out = np.zeros(length, dtype=float)
                for i in range(length):
                    start = indptr[i]
                    end = indptr[i + 1]
                    if end > start:
                        out[i] = float(data[start])
                return out

            row_vals = _csr_vector_payload_to_dense(payloads[0], nnz)
            col_vals = _csr_vector_payload_to_dense(payloads[1], nnz)

            shape = tuple(len(tfsdp.sequences) for tfsdp in tfsdps)
            row_utilities = lil_array(shape)
            column_utilities = lil_array(shape)
            for k in range(nnz):
                i = int(coords[k, 0])
                j = int(coords[k, 1])
                row_utilities[i, j] = float(row_vals[k])
                column_utilities[i, j] = float(col_vals[k])

            return cls(tfsdps, [row_utilities.tocsr(), column_utilities.tocsr()])

        # New packed sparse format (persist_openspiel_game_per_agent meta version >= 2).
        if isinstance(utilities, dict) and utilities.get('kind') == 'scipy.sparse.csr':
            from scipy.sparse import csr_array  # type: ignore

            if int(utilities.get('player_count', 2)) != 2:
                raise ValueError('utility is not of a 2-player game')
            if bool(utilities.get('zero_sum', False)):
                raise ValueError('expected general-sum utilities, got zero-sum')

            def _unpack_csr(payload):
                if payload.get('type') != 'csr':
                    raise ValueError('unsupported sparse utility type')
                shape = tuple(payload['shape'])
                data = payload['data']
                indices = payload['indices']
                indptr = payload['indptr']
                return csr_array((data, indices, indptr), shape=shape)

            row = _unpack_csr(utilities['row_utility'])
            col = _unpack_csr(utilities['column_utility'])
            return cls(tfsdps, [row, col])

        # Legacy template-compatible list format.
        shape = tuple(len(tfsdp.sequences) for tfsdp in tfsdps)
        row_utilities = lil_array(shape)
        column_utilities = lil_array(shape)

        for raw_utility in utilities:
            if len(raw_utility['values']) != 2:
                raise ValueError('utility is not of a 2-player game')

            indices = []
            for tfsdp, sequence in zip(tfsdps, raw_utility['sequences']):
                sequence = tuple(sequence)
                indices.append(tfsdp.sequences.index(sequence))
            indices = tuple(indices)

            row_utilities[indices] = raw_utility['values'][0]
            column_utilities[indices] = raw_utility['values'][1]

        return cls(tfsdps, [row_utilities.tocsr(), column_utilities.tocsr()])

    def _verify(self, **kwargs):
        super()._verify(**kwargs)

        if not (
                self.row_utilities.shape
                == self.column_utilities.shape
                == (len(self.row_sequences), len(self.column_sequences))
        ):
            raise ValueError('utilities do not match sequences')

    @property
    def row_tree_form_sequential_decision_process(self):
        return self.tree_form_sequential_decision_processes[0]

    @property
    def column_tree_form_sequential_decision_process(self):
        return self.tree_form_sequential_decision_processes[1]

    @property
    def row_sequences(self):
        return self.row_tree_form_sequential_decision_process.sequences

    @property
    def column_sequences(self):
        return self.column_tree_form_sequential_decision_process.sequences

    @property
    def row_utilities(self):
        return self.utilities[0]

    @property
    def column_utilities(self):
        return self.utilities[1]

    def row_best_response(self, column_strategy):
        best_response = (
            self
            .row_tree_form_sequential_decision_process
            .sequence_form_best_response(self.row_utility(column_strategy))
        )

        return best_response

    def column_best_response(self, row_strategy):
        best_response = (
            self
            .column_tree_form_sequential_decision_process
            .sequence_form_best_response(self.column_utility(row_strategy))
        )

        return best_response

    def serialize(self):
        tfsdps = self.tree_form_sequential_decision_processes
        raw_tfsdps = [tfsdp.to_list() for tfsdp in tfsdps]
        raw_utilities = []
        abs_utility_sums = abs(self.row_utilities) + abs(self.column_utilities)

        for indices in zip(*abs_utility_sums.nonzero()):
            sequences = []

            for tfsdp, index in zip(tfsdps, indices):
                sequences.append(tfsdp.sequences[index])

            row_value = self.row_utilities[indices].item()
            column_value = self.column_utilities[indices].item()
            values = row_value, column_value

            raw_utilities.append({'sequences': sequences, 'values': values})

        return {
            'tree_form_sequential_decision_processes': raw_tfsdps,
            'utilities': raw_utilities,
        }


@dataclass
class TwoPlayerZeroSumExtensiveFormGame(
        TwoPlayerZeroSumGame,
        TwoPlayerExtensiveFormGame,
):
    """Two-player zero-sum (2p0s) extensive-form game (EFG).

    The utility matrix is from the viewpoint of the row player.
    """

    @classmethod
    def deserialize(cls, raw_data):
        tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        utilities = raw_data['utilities']

        # New packed sparse format (persist_openspiel_game_per_agent meta version >= 2).
        if isinstance(utilities, dict) and utilities.get('kind') == 'scipy.sparse.csr':
            from scipy.sparse import csr_array  # type: ignore

            if int(utilities.get('player_count', 2)) != 2:
                raise ValueError('utility is not of a 2-player game')
            if not bool(utilities.get('zero_sum', True)):
                raise ValueError('expected zero-sum utilities, got general-sum')

            payload = utilities['utility']
            if payload.get('type') != 'csr':
                raise ValueError('unsupported sparse utility type')
            shape = tuple(payload['shape'])
            data = payload['data']
            indices = payload['indices']
            indptr = payload['indptr']
            u = csr_array((data, indices, indptr), shape=shape)
            return cls(tfsdps, u)

        # Legacy template-compatible list format.
        shape = tuple(len(tfsdp.sequences) for tfsdp in tfsdps)
        u_lil = lil_array(shape)

        for raw_utility in utilities:
            indices = []
            for tfsdp, sequence in zip(tfsdps, raw_utility['sequences']):
                sequence = tuple(sequence)
                indices.append(tfsdp.sequences.index(sequence))
            indices = tuple(indices)
            u_lil[indices] = raw_utility['value']

        return cls(tfsdps, u_lil.tocsr())

    @property
    def row_utilities(self):
        return self.utilities

    def serialize(self):
        tfsdps = self.tree_form_sequential_decision_processes
        raw_tfsdps = [tfsdp.to_list() for tfsdp in tfsdps]
        raw_utilities = []

        for indices in zip(*self.utilities.nonzero()):
            sequences = []

            for tfsdp, index in zip(tfsdps, indices):
                sequences.append(tfsdp.sequences[index])

            value = self.utilities[indices].item()

            raw_utilities.append({'sequences': sequences, 'value': value})

        return {
            'tree_form_sequential_decision_processes': raw_tfsdps,
            'utilities': raw_utilities,
        }


@dataclass
class MultiPlayerExtensiveFormGame(ExtensiveFormGame):
    """Multi-player (n-player) general-sum extensive-form game (EFG).

    This class mirrors the 2-player `TwoPlayerExtensiveFormGame` API but stores
    utilities in a sparse "profile list" representation:
    - `coords`: (nnz, player_count) integer indices of per-player sequences
    - `values`: list of length player_count; each entry is a dense float vector
      of length nnz containing that player's payoff at each profile row.

    For a fixed opponent strategy profile x_{-i}, the sequence-form utility
    vector for player i is:
      u_i[s_i] = sum_{s_{-i}} U_i[s_i, s_{-i}] * prod_{j!=i} x_j[s_j]
    which we compute efficiently via `np.bincount` over sparse profiles.
    """

    @classmethod
    def deserialize(cls, raw_data):
        tfsdps = TreeFormSequentialDecisionProcess.deserialize_all(
            raw_data['tree_form_sequential_decision_processes'],
        )
        utilities = raw_data['utilities']
        player_count = len(tfsdps)

        def _csr_vector_payload_to_dense(payload: dict[str, Any], length: int) -> np.ndarray:
            """Unpack an (nnz x 1) CSR payload into a dense length-nnz vector."""
            if payload.get('type') != 'csr':
                raise ValueError('unsupported sparse utility type')
            shape = tuple(payload['shape'])
            if shape != (length, 1):
                raise ValueError('unexpected sparse vector shape')
            indptr = np.asarray(payload['indptr'])
            data = np.asarray(payload['data'])
            # Each row has 0 or 1 stored entries (column index is always 0).
            row_nnz = (indptr[1:] - indptr[:-1]).astype(np.int64, copy=False)
            out = np.zeros(length, dtype=float)
            mask = row_nnz > 0
            if mask.any():
                positions = indptr[:-1][mask].astype(np.int64, copy=False)
                out[mask] = data[positions].astype(float, copy=False)
            return out

        # Packed sparse profiles with per-player value vectors (recommended).
        if isinstance(utilities, dict) and utilities.get('kind') == 'scipy.sparse.profile_per_player':
            if int(utilities.get('player_count', player_count)) != player_count:
                raise ValueError('utilities player_count does not match tfsdps')
            if bool(utilities.get('zero_sum', False)):
                raise ValueError('expected general-sum utilities, got zero-sum')

            coords = np.asarray(utilities['coords'], dtype=np.int64)
            nnz = int(coords.shape[0])
            payloads = list(utilities['values'])
            if len(payloads) != player_count:
                raise ValueError('utilities values do not match player_count')
            values = [
                _csr_vector_payload_to_dense(payloads[p], nnz)
                for p in range(player_count)
            ]
            return cls(tfsdps, {'coords': coords, 'values': values})

        # Legacy/raw list format (template compatible).
        if isinstance(utilities, list):
            seq_index = [
                {seq: i for i, seq in enumerate(tfsdps[p].sequences)}
                for p in range(player_count)
            ]
            nnz = len(utilities)
            coords = np.empty((nnz, player_count), dtype=np.int64)
            values = [np.zeros(nnz, dtype=float) for _ in range(player_count)]
            for k, raw_utility in enumerate(utilities):
                seqs = raw_utility.get('sequences')
                vals = raw_utility.get('values')
                if seqs is None or vals is None:
                    raise ValueError('expected raw utility entries with sequences/values')
                if len(seqs) != player_count or len(vals) != player_count:
                    raise ValueError('raw utility entry does not match player_count')
                for p in range(player_count):
                    seq = tuple(seqs[p])
                    coords[k, p] = int(seq_index[p][seq])
                    values[p][k] = float(vals[p])
            return cls(tfsdps, {'coords': coords, 'values': values})

        raise ValueError('unsupported utilities format for MultiPlayerExtensiveFormGame')

    def _verify(self, **kwargs):
        super()._verify(**kwargs)

        coords = self.utilities.get('coords') if isinstance(self.utilities, dict) else None
        values = self.utilities.get('values') if isinstance(self.utilities, dict) else None
        if coords is None or values is None:
            raise ValueError('missing utilities coords/values')
        if int(np.asarray(coords).shape[1]) != self.player_count:
            raise ValueError('coords do not match player_count')
        if len(values) != self.player_count:
            raise ValueError('values do not match player_count')

    def dimension(self, player):
        if not (0 <= int(player) < self.player_count):
            raise ValueError(f'Player {player} does not exist')
        return len(self.tree_form_sequential_decision_processes[int(player)].sequences)

    @property
    def utility_coords(self) -> np.ndarray:
        return np.asarray(self.utilities['coords'], dtype=np.int64)

    @property
    def utility_values(self) -> list[np.ndarray]:
        return list(self.utilities['values'])

    def utility(self, player, *opponent_strategies):
        player = int(player)
        if len(opponent_strategies) != self.player_count - 1:
            raise ValueError('expected opponent strategies for all other players')
        full = list(opponent_strategies)
        full.insert(player, None)

        coords = self.utility_coords
        nnz = int(coords.shape[0])
        reach = np.ones(nnz, dtype=float)
        for p in range(self.player_count):
            if p == player:
                continue
            s = np.asarray(full[p], dtype=float)
            reach *= s[coords[:, p]]

        weights = self.utility_values[player] * reach
        dim = self.dimension(player)
        return np.bincount(
            coords[:, player],
            weights=weights,
            minlength=dim,
        ).astype(float, copy=False)

    def value(self, player, *strategies):
        player = int(player)
        if len(strategies) != self.player_count:
            raise ValueError('expected strategies for all players')
        coords = self.utility_coords
        nnz = int(coords.shape[0])
        reach = np.ones(nnz, dtype=float)
        for p in range(self.player_count):
            s = np.asarray(strategies[p], dtype=float)
            reach *= s[coords[:, p]]
        return float(np.sum(self.utility_values[player] * reach))

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        player = int(player)
        utility = self.utility(player, *opponent_strategies)
        tfsdp = self.tree_form_sequential_decision_processes[player]
        return tfsdp.sequence_form_best_response(utility)

    def serialize(self):
        raise NotImplementedError


@dataclass
class SymmetrizedGame(Game):
    """Symmetrized game.

    Each player optimizes over the cartesian product of probability
    simplices.
    """

    game: Any

    @property
    def player_count(self):
        return self.game.player_count

    def dimension(self, player):
        return sum(self.game.dimensions)

    def utility(self, player, *opponent_strategies):
        strategies = []

        for opponent_strategy in opponent_strategies:
            strategies.append(split(opponent_strategy, self.game.dimensions))

        strategies.insert(player, None)

        utilities = [0] * self.player_count

        for permutation in permutations(range(self.player_count)):
            utilities[permutation.index(player)] += self.game.utility(
                permutation.index(player),
                *(
                    strategies[permutation[i]][i]
                    for i in range(self.player_count)
                    if permutation[i] != player
                ),
            )

        utility = np.concatenate(utilities)
        utility /= factorial(self.player_count)

        return utility

    def value(self, player, *strategies):
        strategies = list(strategies)

        for i in range(self.player_count):
            strategies[i] = split(strategies[i], self.game.dimensions)

        value = 0

        for permutation in permutations(range(self.player_count)):
            value += self.game.value(
                permutation.index(player),
                *(
                    strategies[permutation[i]][i]
                    for i in range(self.player_count)
                ),
            )

        value /= factorial(self.player_count)

        return value

    def correlated_value(self, player, *strategies):
        raise NotImplementedError

    def best_response(self, player, *opponent_strategies):
        raise NotImplementedError


class ExtensiveFormGame2(ABC):
    """Extensive-form game (EFG)."""

    @dataclass(frozen=True)
    class State:
        """State of an extensive-form game."""

        @property
        @abstractmethod
        def utilities(self):
            pass

        @property
        @abstractmethod
        def chance_action_probabilities(self):
            pass

        @property
        @abstractmethod
        def actions(self):
            pass

        @property
        @abstractmethod
        def infoset(self):
            pass

        @property
        @abstractmethod
        def player(self):
            pass

        @abstractmethod
        def is_terminal(self):
            pass

        @abstractmethod
        def is_chance(self):
            pass

        @abstractmethod
        def utility(self, player):
            pass

        @abstractmethod
        def apply(self, action):
            pass

    @property
    @abstractmethod
    def players(self):
        pass

    @property
    @abstractmethod
    def initial_state(self):
        pass

    def values(self, strategy_profile, state=None):
        if state is None:
            values = self.values(strategy_profile, self.initial_state)
        elif state.is_terminal():
            values = state.utilities
        else:
            if state.is_chance():
                actions, probabilities = zip(
                    *state.chance_action_probabilities,
                )
            else:
                actions = state.actions
                probabilities = strategy_profile(state)

            values = 0

            for action, probability in zip(actions, probabilities):
                values += (
                    probability
                    * self.values(strategy_profile, state.apply(action))
                )

        return values

    def best_response_value(self, player, strategy_profile):
        states = defaultdict(list)
        counterfactual_reach_probabilities = {}

        def dfs(state, counterfactual_reach_probability):
            counterfactual_reach_probabilities[state] = (
                counterfactual_reach_probability
            )

            if state.is_terminal():
                return

            if not state.is_chance():
                states[state.infoset].append(state)

            if state.is_chance() or state.player != player:
                if state.is_chance():
                    actions, probabilities = zip(
                        *state.chance_action_probabilities,
                    )
                else:
                    actions = state.actions
                    probabilities = strategy_profile(state)

                for action, probability in zip(actions, probabilities):
                    dfs(
                        state.apply(action),
                        probability * counterfactual_reach_probability,
                    )
            else:
                for action in state.actions:
                    dfs(state.apply(action), counterfactual_reach_probability)

        dfs(self.initial_state, 1)

        @cache
        def solve(state):
            if state.is_terminal():
                value = state.utility(player)
            elif state.is_chance() or state.player != player:
                if state.is_chance():
                    actions, probabilities = zip(
                        *state.chance_action_probabilities,
                    )
                else:
                    actions = state.actions
                    probabilities = strategy_profile(state)

                value = 0

                for action, probability in zip(actions, probabilities):
                    value += probability * solve(state.apply(action))
            else:
                value = solve2(state.infoset)

            return value

        @cache
        def solve2(infoset):
            values = defaultdict(int)

            for state in states[infoset]:
                weight = counterfactual_reach_probabilities[state]

                for i, action in enumerate(state.actions):
                    values[i] += weight * solve(state.apply(action))

            return max(values.values())

        return solve(self.initial_state)

    def nash_gap(self, strategy_profile):
        gap = 0

        for player, value in zip(self.players, self.values(strategy_profile)):
            best_response_value = self.best_response_value(
                player,
                strategy_profile,
            )
            gap += best_response_value - value

        return gap
